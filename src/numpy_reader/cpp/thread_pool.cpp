// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// cuda stuff
#include <cuda.h>
#include <cuda_runtime.h>

// other stuff
#include <cstdlib>
#include <utility>
#include "thread_pool.h"

ThreadPool::ThreadPool(int num_thread, int device_id, bool set_affinity)
    : threads_(num_thread), running_(true), work_complete_(true), adding_work_(false)
    , active_threads_(0) {
      
  if (num_thread <= 0) {
    throw std::runtime_error("Thread pool must have non-zero size");
  }

  // Start the threads in the main loop
  for (int i = 0; i < num_thread; ++i) {
    threads_[i] = std::thread(std::bind(&ThreadPool::ThreadMain, this, i, device_id, set_affinity));
  }
  tl_errors_.resize(num_thread);
}

ThreadPool::~ThreadPool() {
  WaitForWork(false);

  std::unique_lock<std::mutex> lock(mutex_);
  running_ = false;
  condition_.notify_all();
  lock.unlock();

  for (auto &thread : threads_) {
    thread.join();
  }
}

void ThreadPool::AddWork(Work work, int64_t priority, bool finished_adding_work) {
  std::lock_guard<std::mutex> lock(mutex_);
  work_queue_.push({priority, std::move(work)});
  work_complete_ = false;
  adding_work_ = !finished_adding_work;
}

void ThreadPool::DoWorkWithID(Work work, int64_t priority) {
  AddWork(std::move(work), priority, true);
  // Signal a thread to complete the work
  condition_.notify_one();
}

// Blocks until all work issued to the thread pool is complete
void ThreadPool::WaitForWork(bool checkForErrors) {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_.wait(lock, [this] { return this->work_complete_; });

  if (checkForErrors) {
    // Check for errors
    for (size_t i = 0; i < threads_.size(); ++i) {
      if (!tl_errors_[i].empty()) {
        // Throw the first error that occurred
	std::string error = "Error in thread " + std::to_string(i) + ": " + tl_errors_[i].front();
        tl_errors_[i].pop();
        throw std::runtime_error(error);
      }
    }
  }
}

void ThreadPool::RunAll(bool wait) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    adding_work_ = false;
  }
  condition_.notify_one();  // other threads will be waken up if needed
  if (wait) {
    WaitForWork();
  }
}

int ThreadPool::size() const {
  return threads_.size();
}

std::vector<std::thread::id> ThreadPool::GetThreadIds() const {
  std::vector<std::thread::id> tids;
  tids.reserve(threads_.size());
  for (const auto &thread : threads_)
    tids.emplace_back(thread.get_id());
  return tids;
}


void ThreadPool::ThreadMain(int thread_id, int device_id, bool set_affinity) {
  // set device
  cudaSetDevice(device_id);

  while (running_) {
    // Block on the condition to wait for work
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return !running_ || (!work_queue_.empty() && !adding_work_); });
    // If we're no longer running, exit the run loop
    if (!running_) break;

    // Get work from the queue & mark
    // this thread as active
    Work work = std::move(work_queue_.top().second);
    work_queue_.pop();
    bool should_wake_next = !work_queue_.empty();
    ++active_threads_;

    // Unlock the lock
    lock.unlock();

    if (should_wake_next) {
      condition_.notify_one();
    }

    // If an error occurs, we save it in tl_errors_. When
    // WaitForWork is called, we will check for any errors
    // in the threads and return an error if one occured.
    try {
      work(thread_id);
    } catch (std::exception &e) {
      lock.lock();
      tl_errors_[thread_id].push(e.what());
      lock.unlock();
    } catch (...) {
      lock.lock();
      tl_errors_[thread_id].push("Caught unknown exception");
      lock.unlock();
    }

    // Mark this thread as idle & check for complete work
    lock.lock();
    --active_threads_;
    if (work_queue_.empty() && active_threads_ == 0) {
      work_complete_ = true;
      completed_.notify_one();
    }
  }
}
