"""Reassign intact loss jobs within one update, without changing their weights."""

import statistics


def job_tokens(job):
    rows, length, _ = job["tensor_shape"]
    return rows * length


def partition_by_time(workers, seconds_per_token):
    known = [value for value in seconds_per_token if value is not None]
    default = statistics.median(known) if known else 1.
    rates = [default if value is None else value for value in seconds_per_token]
    jobs = sorted((job for worker in workers for job in worker), key=job_tokens, reverse=True)
    world = len(workers)
    # Every rank needs a real final backward for DDP's one synchronization.
    assert len(jobs) >= world
    bins, tokens = [[] for _ in workers], [0] * world
    initial = sorted(range(world), key=rates.__getitem__)
    for index, job in enumerate(jobs):
        size = job_tokens(job)
        owner = initial[index] if index < world else min(
            range(world), key=lambda rank: (tokens[rank] + size) * rates[rank])
        bins[owner].append(job)
        tokens[owner] += size
    for batch in bins:
        # A short genuine job is sufficient to synchronize all accumulated grads.
        last = min(range(len(batch)), key=lambda i: job_tokens(batch[i]))
        batch.append(batch.pop(last))
    return bins
