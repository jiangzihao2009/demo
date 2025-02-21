#include <condition_variable>
#include <list>
#include <mutex>

struct ObjectPoolConfig {
  int default_object_count = 10;
};

template <typename T> class ObjectPool {
public:
  class ObjectRef {
  public:
    ObjectRef(ObjectPool &pool, std::unique_ptr<T> &&obj)
        : pool_(pool), obj_(std::move(obj)) {}
    ~ObjectRef() {
      if (Valid()) {
        pool_.Recycle(std::move(obj_));
      }
    }
    T *operator->() { return obj_.get(); }
    bool Valid() { return (obj_ != nullptr); }

  private:
    ObjectPool &pool_;
    std::unique_ptr<T> obj_;
  };
  ObjectPool(const ObjectPoolConfig &config) : config_(config) {}
  template <typename... Args> void Initialize(Args... args) {
    for (int i = 0; i < config_.default_object_count; i++) {
      auto obj = std::make_unique<T>(args...);
      pool_.emplace_back(std::move(obj));
    }
  }
  ObjectRef TryApply() {
    std::unique_lock<std::mutex> lock(mtx_);
    if (pool_.empty()) {
      return ObjectRef(*this, nullptr);
    }
    return GetOne();
  }
  bool Empty() {
    std::lock_guard<std::mutex> lock(mtx_);
    return pool_.empty();
  }
  ObjectRef Apply() {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this]() { return !pool_.empty(); });
    return GetOne();
  }
  int Count() {
    std::unique_lock<std::mutex> lock(mtx_);
    return pool_.size();
  }
  void Recycle(std::unique_ptr<T> &&obj) {
    if (obj == nullptr) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mtx_);
      pool_.emplace_back(std::move(obj));
    }
    cv_.notify_one();
  }

private:
  ObjectRef GetOne() {
    auto last = std::move(pool_.back());
    pool_.pop_back();
    return ObjectRef(*this, std::move(last));
  }
  const ObjectPoolConfig config_;
  std::mutex mtx_;
  std::condition_variable cv_;
  std::list<std::unique_ptr<T>> pool_;
};
