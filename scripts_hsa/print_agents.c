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
  bool host_accessible;
  size_t size;
  size_t max_alloc_size;
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
  hsa_amd_coherency_type_t coherent;
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
  check(hsa_region_get_info(region, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED, &regions->regions[regions->count].info.alloc_allowed));
  check(hsa_region_get_info(region, (hsa_region_info_t)HSA_AMD_REGION_INFO_HOST_ACCESSIBLE, &regions->regions[regions->count].info.host_accessible));
  check(hsa_region_get_info(region, HSA_REGION_INFO_ALLOC_MAX_SIZE, &regions->regions[regions->count].info.max_alloc_size));
  check(hsa_region_get_info(region, HSA_REGION_INFO_SIZE, &regions->regions[regions->count].info.size));
  if (regions->regions[regions->count].info.segment == HSA_REGION_SEGMENT_GLOBAL) {
    check(hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &regions->regions[regions->count].info.global_flags));
  } else {
    regions->regions[regions->count].info.global_flags = UINT32_MAX;
  }
  regions->regions[regions->count].region = region;
  regions->count++;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t get_agent_callback(hsa_agent_t agent, void *data) {
  Agents *agents = (Agents *)data;
  check(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &agents->agents[agents->count].info.type));
  if (agents->agents[agents->count].info.type != HSA_DEVICE_TYPE_GPU) {
    return HSA_STATUS_SUCCESS;
  }
  check(hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_UUID, agents->agents[agents->count].info.name));
  check(hsa_amd_coherency_set_type(agent, HSA_AMD_COHERENCY_TYPE_COHERENT));
  check(hsa_amd_coherency_get_type(agent, &agents->agents[agents->count].info.coherent));
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
    printf("  Coherent: %s\n", (info.coherent == HSA_AMD_COHERENCY_TYPE_COHERENT) ? "yes" : "no");
    printf("  Regions:\n");
    Regions regions = agents.agents[iagent].regions;
    for (int iregion = 0; iregion < regions.count; ++iregion) {
      RegionInfo reg_info = regions.regions[iregion].info;
      printf("    Region # %d:\n", iregion);
      printf("      Handle: %lX\n", regions.regions[iregion].region.handle);
      printf("      Accesible by host: %s\n", reg_info.host_accessible ? "yes" : "no");
      switch (reg_info.segment) {
      case HSA_REGION_SEGMENT_GLOBAL:
        printf("      Segment: global\n");
        printf("        Coarse-grained: ");
        if ((reg_info.global_flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) == HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) {
          printf("yes\n");
        } else {
          printf("no\n");
        }
        printf("        Fine-grained: ");
        if ((reg_info.global_flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) == HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) {
          printf("yes\n");
        } else {
          printf("no\n");
        }
        printf("        Kernel arguments: ");
        if ((reg_info.global_flags & HSA_REGION_GLOBAL_FLAG_KERNARG) == HSA_REGION_GLOBAL_FLAG_KERNARG) {
          printf("yes\n");
        } else {
          printf("no\n");
        }
        break;
      case HSA_REGION_SEGMENT_READONLY:
        printf("      Segment: read-only\n");
        break;
      case HSA_REGION_SEGMENT_PRIVATE:
        printf("      Segment: private\n");
        break;
      case HSA_REGION_SEGMENT_GROUP:
        printf("      Segment: group\n");
        break;
      case HSA_REGION_SEGMENT_KERNARG:
        printf("      Segment: kernel arguments\n");
        break;
      default:
        printf("      Segment: other\n");
        break;
      }
      printf("      Allocation allowed: %s\n", reg_info.alloc_allowed ? "yes" : "no");
      printf("      Size: %lu MiB\n", reg_info.size >> 20);
      printf("      Max allocation size: %lu MiB\n", reg_info.max_alloc_size >> 20);
    }
  }

  check(hsa_shut_down());
  return EXIT_SUCCESS;
}