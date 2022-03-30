#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define check(expr)                                                                                                                                            \
  do {                                                                                                                                                         \
    hsa_status_t err = expr;                                                                                                                                   \
    if (err != HSA_STATUS_SUCCESS) {                                                                                                                           \
      const char *msg;                                                                                                                                         \
      hsa_status_string(err, &msg);                                                                                                                            \
      fprintf(stderr, "Error (file %s, line %d): %s\n", __FILE__, __LINE__, msg);                                                                              \
      exit(EXIT_FAILURE);                                                                                                                                      \
    }                                                                                                                                                          \
  } while (0)

typedef struct RegionInfo_s {
  hsa_region_segment_t segment;
  bool alloc_allowed;
  size_t size;
} RegionInfo;

typedef struct Region_s {
  hsa_region_t region;
  RegionInfo info;
} Region;

typedef struct Regions_s {
  Region regions[128];
  int count;
} Regions;

typedef struct AgentInfo_s {
  char name[64];
  hsa_device_type_t type;
} AgentInfo;

typedef struct Agent_s {
  hsa_agent_t agent;
  AgentInfo info;
  Regions regions;
} Agent;

typedef struct Agents_s {
  Agent agents[16];
  int count;
} Agents;

hsa_status_t get_gpu_region_callback(hsa_region_t region, void *data) {
  hsa_region_t *stored_region = data;
  hsa_region_segment_t segment;
  bool alloc_allowed;
  bool host_accessible;
  size_t region_size;
  check(hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment));
  if (segment != HSA_REGION_SEGMENT_GLOBAL)
    return HSA_STATUS_SUCCESS;
  check(hsa_region_get_info(region, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed));
  if (!alloc_allowed)
    return HSA_STATUS_SUCCESS;
  check(hsa_region_get_info(region, HSA_AMD_REGION_INFO_HOST_ACCESSIBLE, &host_accessible));
  if (host_accessible) return HSA_STATUS_SUCCESS;
  *stored_region = region;
  return HSA_STATUS_INFO_BREAK;
}

typedef struct AgentData_s {
  hsa_agent_t agent;
  hsa_device_type_t type;
  int index;
  int counter;
} AgentData;

hsa_status_t get_agent_callback(hsa_agent_t agent, void *data) {
  AgentData *agent_data = data;

  hsa_device_type_t device_type;
  check(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type));
  if (device_type == agent_data->type && agent_data->counter == agent_data->index) {
    agent_data->agent = agent;
    return HSA_STATUS_INFO_BREAK;
  }
  agent_data->counter++;
  return HSA_STATUS_SUCCESS;
}

hsa_agent_t get_agent(hsa_device_type_t type, int index) {
  AgentData agent_data = {.type = type, .index = index};
  check(hsa_iterate_agents(get_agent_callback, &agent_data));
  return agent_data.agent;
}

int main() {
  printf("Initialising HSA...");
  check(hsa_init());
  printf(" done\n");

  hsa_agent_t cpu_agent = get_agent(HSA_DEVICE_TYPE_CPU, 0);
  hsa_agent_t gpu_agent = get_agent(HSA_DEVICE_TYPE_GPU, 0);

  Region host_region = cpu_agent.regions.regions[0];
  Region device_region = gpu_agent.regions.regions[0];

  double *A_h = NULL;
  double *A_h_locked = NULL;
  double *A = NULL;
  size_t nx = 10, ny = 11;

  A_h = malloc(nx * ny * sizeof(double));
  check(hsa_memory_allocate(device_region.region, sizeof(double) * nx * ny, (void **)&A));

  for (size_t iy = 0, idx = 0; iy < ny; ++iy) {
    for (size_t ix = 0; ix < nx; ++ix, ++idx) {
      A_h[idx] = ((double)rand()) / RAND_MAX;
    }
  }

  printf("\n  Matrix (before copy to device):\n");
  print_matrix(A_h, nx, ny);

  check(hsa_memory_copy(A, A_h, nx * ny * sizeof(double)));
  memset(A_h, 0, nx * ny * sizeof(double));

  printf("\n  Matrix (after memset):\n");
  print_matrix(A_h, nx, ny);

  hsa_agent_t lock_agents[] = {gpu_agent.agent, cpu_agent.agent};

  check(hsa_amd_memory_lock(A_h, nx * ny * sizeof(double), lock_agents, 2, (void **)&A_h_locked));

  hsa_signal_t completion_signal;

  check(hsa_signal_create(1, 1, &cpu_agent.agent, &completion_signal));

  check(hsa_amd_memory_async_copy(A_h_locked, cpu_agent.agent, A, gpu_agent.agent, nx * ny * sizeof(double), 0, NULL, completion_signal));

  check(hsa_signal_wait_scacquire(completion_signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE));

  printf("\n  Matrix (after copy to host):\n");
  print_matrix(A_h, nx, ny);

  check(hsa_amd_memory_unlock(A_h_locked));

  check(hsa_memory_free(A));
  free(A_h);

  check(hsa_shut_down());
  return EXIT_SUCCESS;
}