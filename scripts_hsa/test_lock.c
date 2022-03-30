#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define check(expr)                                                                                                    \
  do {                                                                                                                 \
    hsa_status_t err = expr;                                                                                           \
    if (err != HSA_STATUS_SUCCESS) {                                                                                   \
      const char *msg;                                                                                                 \
      hsa_status_string(err, &msg);                                                                                    \
      fprintf(stderr, "Error (file %s, line %d): %s\n", __FILE__, __LINE__, msg);                                      \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

static void print_matrix(double *A, size_t nx, size_t ny) {
  for (size_t iy = 0, idx = 0; iy < ny; ++iy) {
    printf("      ");
    for (size_t ix = 0; ix < nx; ++ix, ++idx) {
      printf("%.3f  ", A[idx]);
    }
    printf("\n");
  }
}

typedef struct RegionInfo_s {
  hsa_region_segment_t segment;
  bool alloc_allowed;
  size_t size;
  uint32_t global_flags;
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
  Agent agents[128];
  int count;
} Agents;

hsa_status_t get_region_callback(hsa_region_t region, void *data) {
  Regions *regions = (Regions *)data;
  check(hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &regions->regions[regions->count].info.segment));
  check(hsa_region_get_info(region, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED,
                            &regions->regions[regions->count].info.alloc_allowed));
  check(hsa_region_get_info(region, HSA_REGION_INFO_ALLOC_MAX_SIZE, &regions->regions[regions->count].info.size));
  if (regions->regions[regions->count].info.segment == HSA_REGION_SEGMENT_GLOBAL) {
    check(
        hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &regions->regions[regions->count].info.global_flags));
  } else {
    regions->regions[regions->count].info.global_flags = UINT32_MAX;
  }
  regions->regions[regions->count].region = region;
  regions->count++;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t get_agent_callback(hsa_agent_t agent, void *data) {
  Agents *agents = (Agents *)data;
  check(hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agents->agents[agents->count].info.name));
  check(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &agents->agents[agents->count].info.type));
  agents->agents[agents->count].agent = agent;
  // get supported memory regions for agent
  Regions regions = {};
  check(hsa_agent_iterate_regions(agent, get_region_callback, &regions));
  agents->agents[agents->count].regions = regions;
  agents->count++;
  return HSA_STATUS_SUCCESS;
}

Agents get_agents() {
  Agents agents = {};
  check(hsa_iterate_agents(get_agent_callback, &agents));
  return agents;
}

int main() {
  printf("Initialising HSA...");
  check(hsa_init());
  printf(" done\n");

  Agents agents = get_agents();

  Agent cpu_agent = agents.agents[0];
  Agent gpu_agent = agents.agents[2];

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

  check(hsa_amd_memory_async_copy(A_h_locked, cpu_agent.agent, A, gpu_agent.agent, nx * ny * sizeof(double), 0, NULL,
                                  completion_signal));

  check(hsa_signal_wait_scacquire(completion_signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE));

  printf("\n  Matrix (after copy to host):\n");
  print_matrix(A_h, nx, ny);

  check(hsa_amd_memory_unlock(A_h_locked));

  check(hsa_memory_free(A));
  free(A_h);

  check(hsa_shut_down());
  return EXIT_SUCCESS;
}