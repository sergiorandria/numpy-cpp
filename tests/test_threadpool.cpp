/**
 * @file test_threadpool.cpp
 * @brief Tests for np::ThreadPool work-stealing.
 */
#include <atomic>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "np/threadpool.hpp"
#include "test_util.hpp"

int main()
{
    // Basic submit + future
    {
        np::ThreadPool pool(4);
        auto fut = pool.submit([] { return 42; });
        test::check(fut.get() == 42, "submit future");
    }

    // Parallel increment with atomics
    {
        np::ThreadPool pool(4);
        std::atomic<int> c{0};
        const int n = 10000;
        pool.parallel_for(0, n, [&](std::size_t) { c.fetch_add(1, std::memory_order_relaxed); });
        test::check(c.load() == n, "parallel_for count");
    }

    // Work stealing: many small tasks, verify all executed
    {
        np::ThreadPool pool(8);
        std::atomic<int> done{0};
        const int tasks = 200;
        std::vector<std::future<int>> futs;
        futs.reserve(tasks);
        for (int i = 0; i < tasks; ++i)
        {
            futs.push_back(pool.submit([i, &done] {
                done.fetch_add(1, std::memory_order_relaxed);
                return i * 2;
            }));
        }
        int sum = 0;
        for (auto &f : futs)
        {
            sum += f.get();
        }
        test::check(done.load() == tasks, "many tasks count");
        test::check(sum == tasks * (tasks - 1), "many tasks sum");
    }

    // parallel_for with chunk and result aggregation
    {
        np::ThreadPool pool(0); // auto hardware_concurrency
        const std::size_t n = 100000;
        std::vector<int> data(n, 1);
        std::atomic<long> total{0};
        pool.parallel_for(0, n, [&](std::size_t i) { total.fetch_add(data[i], std::memory_order_relaxed); });
        test::check(total.load() == static_cast<long>(n), "parallel_for large");
    }

    // Global pool convenience
    {
        std::atomic<int> x{0};
        np::parallel_for(0, 1000, [&](std::size_t) { x.fetch_add(1); });
        test::check(x.load() == 1000, "global parallel_for");
    }

    // ThreadPool size and wait/shutdown
    {
        np::ThreadPool pool(2);
        test::check(pool.size() == 2, "pool size");
        for (int i = 0; i < 10; ++i)
        {
            pool.enqueue([] { std::this_thread::sleep_for(std::chrono::microseconds(10)); });
        }
        pool.wait();
        test::check(true, "wait completes");
        pool.shutdown();
        test::check(true, "shutdown completes");
    }

    // WorkStealingDeque direct test
    {
        np::detail::WorkStealingDeque<int> q;
        q.push_bottom(1);
        q.push_bottom(2);
        auto v = q.pop_bottom();
        test::check(v && *v == 2, "pop_bottom LIFO");
        auto s = q.steal();
        test::check(s && *s == 1, "steal FIFO");
        test::check(q.empty(), "empty after pop/steal");
    }

    // wait() observes in-flight tasks, not just empty queues: enqueue slow
    // tasks that sleep while already popped, then wait() must see all of
    // them done (the old queues-empty-only wait could return early).
    {
        np::ThreadPool pool(4);
        std::atomic<int> done{0};
        const int tasks = 16;
        for (int i = 0; i < tasks; ++i)
        {
            pool.enqueue([&done] {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                done.fetch_add(1, std::memory_order_relaxed);
            });
        }
        pool.wait();
        test::check(done.load() == tasks, "wait sees in-flight tasks");
    }

    // parallel_for propagates chunk exceptions instead of hanging.
    {
        np::ThreadPool pool(4);
        bool threw = false;
        try
        {
            pool.parallel_for(0, 1000, [](std::size_t i) {
                if (i == 500)
                    throw std::runtime_error("boom");
            });
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        test::check(threw, "parallel_for rethrows");
    }

    // A throwing fire-and-forget task is suppressed, pool stays usable.
    {
        np::ThreadPool pool(2);
        pool.enqueue([] { throw std::runtime_error("bg boom"); });
        auto fut = pool.submit([] { return 7; });
        test::check(fut.get() == 7, "pool usable after throw");
        pool.wait();
        test::check(true, "wait after throw");
    }

    return test::failures() ? 1 : 0;
}
