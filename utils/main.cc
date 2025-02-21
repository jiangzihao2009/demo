#include "object_pool.h"
#include <iostream>
#include <string>
#include <thread>

#include <unistd.h>
using namespace std;

class Test {
public:
  Test(const std::string &name, int val) {
    name_ = name;
    val_ = val;
  }
  void Print(const std::string &prefix) const {
    cout << "prefix: " << prefix << ", name: " << name_ << ", value: " << val_
         << endl;
  }

private:
  std::string name_;
  int val_;
};

int main() {
  ObjectPoolConfig cfg;
  ObjectPool<Test> pool(cfg);
  pool.Initialize("poolname", 91);
  thread a([&]() {
    int i = 0;
    while (!pool.Empty() && i < 10) {
      auto t = pool.Apply();
      if (t.Valid()) {
        t->Print("thread1-" + std::to_string(i++));
      }
    }
  });
  thread b([&]() {
    int i = 0;
    while (!pool.Empty() && i < 10) {
      auto t = pool.Apply();
      if (t.Valid()) {
        t->Print("thread2-" + std::to_string(i++));
      }
    }
  });
  a.join();
  b.join();
  cout << "pool count: " << pool.Count() << endl;

  return 0;
}
