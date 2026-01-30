// MORE ON OPENMP

int main () {
    int numt, tid;

    #pragma omp parallel private(tid)
    {
        tid = omp_get_thread_num();
        
        if(tid == 0) {
            numt = omp_get_num_threads();
        }
    }
}

// Alterative way
#pragma omp single nowait
{
    numt = omp_get_num_threads();
} // implicit barrier here




// Use of explicit barriers as synchronisation

int main () {
    int numt, tid;

    #pragma omp parallel shared(numt) private(tid)
    {
        tid = omp_get_thread_num();
        
        if(tid == 0) {
            numt = omp_get_num_threads();
        }

        #pragma omp barrier

        // All threads wait here until thread 0 has set numt
        // Now all threads can safely use numt
        printf("hello world %d of %d\n", tid, numt);
    }
}





// CODE 1

#include <stdio.h>
#include <stdlob.h>
#include <omp.h>

int main() {
    const int N_ACCTS = 8;
    const int N_TXNS = 2000000;
    // calloc - continuous allocation of blocks all init to 0

    long long *balance = (long long *) calloc(N_ACCTS, sizeof(long long));
    if(!balance) return 1;

    // Make the total money non-zero so we can reason about invariants.

    for(int i=0; i < N_ACCTS; i++) {
        balance[i] = 1000;
    }


    omp_lock_t ledger_lock;
    omp_init_lock(&ledger_lock);

    double t0 = omp_get_wtime();

    #pragma omp parallel
    {
        // Thread local seed for random number generation
        unsigned int seed = (unsigned int) omp_get_thread_num();

        #pragma omp for schedule(static)
        for(int k=0; k< N_TXNS; k++) {
            
            int from = (int)(rand_r(&seed) % N_ACCTS);
            int to = (int)(rand_r(&seed) % N_ACCTS);
            if (from == to) continue;

            long long amount = (long long)(rand_r(&seed) % 5);  // small amounts


            // --- critical seciton via an explicit lock ---

            omp_set_lock(&ledger_lock);

            // update shared state safely/
            balance[from] -= amount;
            balance[to] += amount;

            omp_unset_lock(&ledger_lock);
            // --- end critical section ---
        }
    }

    double t1 = omp_get_wtime();

    omp_destroy_lock(&ledger_lock);

    // Check the invariant that total money is conserved
    long long total = 0;
    for(int i=0; i < N_ACCTS; i++) total += balance[i];

    printf("Initial total = %lld\n", N_ACCTS * 1000LL);
    printf("Final total = %lld\n", total);
    printf("Elapsed time = %f seconds\n", t1 - t0);

    free(balance);
    return 0;

}





// EXPLANATION
/*
Book-scribe notes (OpenMP bank-transfer microbenchmark)

- Motivation (why this example?):
    - The example models concurrent access to shared state: multiple threads perform “transfers” between bank accounts.
    - Without synchronization, updates to shared balances can interleave unpredictably, producing non-deterministic results.

- Shared-state concurrency model (what can go wrong?):
    - Each transfer conceptually performs a read–modify–write on two shared memory locations: `balance[from]` and `balance[to]`.
    - A data race occurs when two threads access the same location concurrently and at least one access is a write, with no ordering enforced.
    - Data races break *determinism* and can also break *correctness* (lost updates), even if each individual C statement “looks” simple.

- Program setup (what is being simulated?):
    - `N_ACCTS = 8` accounts (shared array of balances).
    - `N_TXNS = 2,000,000` transfer attempts (large enough to expose contention and overhead).
    - The balances array is allocated with `calloc(N_ACCTS, sizeof(long long))`.
        - Theory note: both `malloc` and `calloc` return a contiguous block of memory; the key difference is that `calloc` zero-initializes.
        - Performance note: whether `calloc` is slower/faster depends on the allocator/OS (zeroing may be optimized); do not assume.
    - All accounts are initialized to the same value (e.g., 1000) so the “total money” is non-zero and easy to reason about.

- Correctness specification (invariants and post-condition):
    - Desired invariant: the sum of all balances should remain constant (“conservation of money”).
    - Each valid transfer subtracts `amount` from one account and adds the same `amount` to another.
    - If every transfer is applied atomically as a pair, then total money is preserved.

- Parallel region mechanics (how the work is distributed):
    - `#pragma omp parallel` creates a team of threads.
    - Each thread has a private RNG seed (here derived from the thread id) to generate pseudo-random transfers.
    - `#pragma omp for schedule(static)` distributes iterations of the transfer loop across threads.
        - Theory note: scheduling affects load balance and reproducibility but does not fix races by itself.

- Why a lock is introduced (synchronization theory):
    - The “transfer” operation updates two shared locations; it must be treated as a single *critical section*.
    - A lock imposes *mutual exclusion*: at most one thread can execute the critical section at a time.
    - In OpenMP, `omp_set_lock` / `omp_unset_lock` provide an ordering that prevents concurrent conflicting updates.

- What the lock guarantees (and what it costs):
    - Correctness: prevents lost updates and makes the invariant check meaningful.
    - Performance: if every iteration takes the same global lock, the parallel loop becomes effectively serialized.
        - This is an instance of *lock contention* and *Amdahl’s Law*: a large serialized fraction limits speedup.

- Timing and validation (engineering practice):
    - `omp_get_wtime()` measures wall-clock time around the parallel loop.
    - After the parallel region, the program sums all balances to validate the conservation invariant.
    - Destroying the lock is good hygiene (release resources; avoids misuse in larger programs).

- Key takeaway (theory-first):
    - Correctness in shared-memory parallelism requires explicit reasoning about atomicity and ordering.
    - A single global lock is the simplest correctness mechanism, but it often eliminates scalability.
    - This example is valuable because it cleanly separates: (1) the *spec* (invariant), (2) the *race*, and (3) the *cost* of naive synchronization.
*/