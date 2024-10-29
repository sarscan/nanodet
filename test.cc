#include <iostream>

class A {
public:
    void x() {
        y();
        std::cout << "this in A " << this << std::endl;
    }

    virtual void y() {
        std::cout << "default behavior" << std::endl;
        std::cout << "this in A " << this << std::endl;
    }
};

class B : public A {
public:
    void y() override {
        std::cout << "Child B behavior" << std::endl;
    }
};

class C : public A {
public:
    void z() {
    }
};

int main() {
    B b;
    b.x();  // 输出: Child B behavior
    std::cout << "b addr " << &b << std::endl;

    C c;
    c.x();  // 输出: default behavior
    std::cout << "c addr " << &c << std::endl;

    A *pb = new B();
    A *pc = new C();
    pb->y();
    std::cout << "pc addr " << pb << std::endl;
    pc->y();
    std::cout << "pc addr " << pc << std::endl;

    return 0;
}
