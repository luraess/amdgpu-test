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
} RegionInfo;

typedef struct Region_s {
  hsa_region_t region;
  RegionInfo info;
} Region;

typedef struct Regions_s {
  Region regions[16];
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

hsa_status_t get_region_callback(hsa_region_t region, void *data) {
  Regions *regions = (Regions *)data;
  check(hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &regions->regions[regions->count].info.segment));
  check(hsa_region_get_info(region, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED,
                            &regions->regions[regions->count].info.alloc_allowed));
  check(hsa_region_get_info(region, HSA_REGION_INFO_ALLOC_MAX_SIZE, &regions->regions[regions->count].info.size));
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

  for (int iagent = 0; iagent < agents.count; ++iagent) {
    AgentInfo info = agents.agents[iagent].info;
    printf("Agent # %d:\n", iagent);
    printf("  Name: %s\n", info.name);
    printf("  Type: %s\n", (info.type == HSA_DEVICE_TYPE_CPU) ? "CPU" : "GPU");
    printf("  Regions:\n");
    Regions regions = agents.agents[iagent].regions;
    for (int iregion = 0; iregion < regions.count; ++iregion) {
      RegionInfo reg_info = regions.regions[iregion].info;
      printf("    Region # %d:\n", iregion);
      printf("      Handle: %lX\n", regions.regions[iregion].region.handle);
      switch (reg_info.segment) {
      case HSA_REGION_SEGMENT_GLOBAL:
        printf("      Segment: global\n");
        break;
      case HSA_REGION_SEGMENT_READONLY:
        printf("      Segment: read-only\n");
        break;
      default:
        printf("      Segment: other\n");
        break;
      }
      printf("      Allocation allowed: %s\n", reg_info.alloc_allowed ? "yes" : "no");
      printf("      Max allocation size: %lu GiB\n", reg_info.size >> 30);
    }
  }

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
  memset(A_h, 0, nx*ny*sizeof(double));

  printf("\n  Matrix (after memset):\n");
  print_matrix(A_h, nx, ny);

  hsa_agent_t lock_agents[] = {gpu_agent.agent, cpu_agent.agent};

  check(hsa_amd_memory_lock(A_h, nx * ny * sizeof(double), lock_agents, 2, (void **)&A_h_locked));

  hsa_signal_t completion_signal;

  check(hsa_signal_create(1, 1, &cpu_agent.agent, &completion_signal));

  check(hsa_amd_memory_async_copy(A_h_locked, cpu_agent.agent, A, gpu_agent.agent, nx*ny*sizeof(double), 0, NULL, completion_signal));

  check(hsa_signal_wait_scacquire(completion_signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE));

  printf("\n  Matrix (after copy to host):\n");
  print_matrix(A_h, nx, ny);

  check(hsa_amd_memory_unlock(A_h_locked));

  check(hsa_memory_free(A));
  free(A_h);

  check(hsa_shut_down());
  return EXIT_SUCCESS;
}