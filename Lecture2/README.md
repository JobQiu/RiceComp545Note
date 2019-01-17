Jan 15, 2019

Some topics in this lecture.

1. Lipschitz continuity
2. Lipschitz gradient continuity
3. Tayler something
4. Convergence rate

https://en.wikipedia.org/wiki/Lipschitz_continuity

###  What is Lipschitz continuity?

<img src="https://ws3.sinaimg.cn/large/006tNc79ly1fz85anfhoqj30gu02et8w.jpg" width="250px"/>

x1, x2 are p-dimensional vectors and function f(x) map them to a value.

This equation means the difference of f(x1) and f(x2) is less than the distance of x1 and x2. The derivation of f(x) has a upper bound, K.

https://en.wikipedia.org/wiki/Lipschitz_continuity#cite_note-3

<img src="https://ws2.sinaimg.cn/large/006tNc79ly1fz86l73vdaj316w04kaat.jpg" width="550px"/>

### Calculate the matrix derivation.

<img src="https://ws4.sinaimg.cn/large/006tNc79ly1fz85plwkk3j31gm0qvnc3.jpg" width="450px"/>

A review of matrix derivative

https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf

### What is singular matrix?

### Some review of gradient descent.


<img src="https://ws4.sinaimg.cn/large/006tNc79ly1fz85wkdru3j31ao0acwfj.jpg" width="350px"/>

Then
1. how to initialize x0?
2. how to update <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Greek_lc_eta.svg/1200px-Greek_lc_eta.svg.png" width="8px"/>?
3. when to terminate iteration?

A1. Randomly initialzie the x0, He initialization, not including in this course.

A2.  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Greek_lc_eta.svg/1200px-Greek_lc_eta.svg.png" width="8px"/> = O(1/t), <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Greek_lc_eta.svg/1200px-Greek_lc_eta.svg.png" width="8px"/> = O(1/sqrt(t)), sometimes, <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Greek_lc_eta.svg/1200px-Greek_lc_eta.svg.png" width="8px"/> becomes too small that might kill the gradient.

A3. time bound, or the value does not change a lot anymore.

### Taylor series review

<img src="https://ws4.sinaimg.cn/large/006tNc79ly1fz862egrf7j30vw03kjrx.jpg" width="450px"/>

### Rate of convergence

<img src="https://upload.wikimedia.org/wikipedia/commons/2/2c/ConvergencePlots.png" width="450px"/>

---

## Summary
