/*
 * cpushim.c - LD_PRELOAD shim that makes sched_getcpu() agree with Wine's
 * processor count when a job runs in a CPU set that does not start at 0.
 *
 * In a 2-CPU Slurm/cgroup cpuset on host CPUs 4-5, Wine 11 reports
 * NumberOfProcessors = 2 but GetCurrentProcessorNumber() returns the raw host
 * ids 4 and 5 (Wine returns sched_getcpu()). Windows guarantees 0..N-1. The
 * .NET 4.8 runtime used by RasProcess.exe indexes per-processor state with that
 * number, and HEC-RAS 6.6 preprocessing then fails intermittently with
 * 0xC0000005 in clr.dll.
 *
 * This shim returns the CPU's index within the process affinity mask
 * (host CPUs 4,5 -> 0,1). The map is built once, on first use; CPUs outside
 * the mask are passed through unchanged. prepare.py preloads it for the
 * Wine processes only.
 *
 * Build: cc -shared -fPIC -O2 -o cpushim.so cpushim.c -ldl
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <sched.h>

static int (*real_getcpu)(void);
static int map_ready;
static short cpu_index[CPU_SETSIZE];

static void build_map(void)
{
    cpu_set_t set;
    int c, n = 0;
    for (c = 0; c < CPU_SETSIZE; c++) cpu_index[c] = -1;
    if (sched_getaffinity(0, sizeof(set), &set) == 0)
        for (c = 0; c < CPU_SETSIZE; c++)
            if (CPU_ISSET(c, &set)) cpu_index[c] = (short)n++;
    __atomic_store_n(&map_ready, 1, __ATOMIC_RELEASE);
}

int sched_getcpu(void)
{
    int cpu;
    if (!real_getcpu) real_getcpu = (int (*)(void))dlsym(RTLD_NEXT, "sched_getcpu");
    if (!__atomic_load_n(&map_ready, __ATOMIC_ACQUIRE)) build_map();
    cpu = real_getcpu ? real_getcpu() : -1;
    if (cpu >= 0 && cpu < CPU_SETSIZE && cpu_index[cpu] >= 0) return cpu_index[cpu];
    return cpu;
}
