## 1. Summary

Add a penalization to control the complexity of w can make the model more general in machine learning. Two commonly used penalities are the l1-norm and l2-norm. Imposing l1-norm have several advantages:
1. It encourages sparse solutions, and a sparse solution enables easier interpretation of the problem in a lower dimension space. Still not clear why sparse is better.
2. l1 constraints can be used for recovering a sparse signal samplesd below the Nyquist rate.
3. In some cases, it leads to improved generalization bounds.


The problem is that `Projections onto l2-ball can be done in linear time while projection onto l1 ball is a more involved task. ` Why? What does projection mean?

Like in the figure, you can project the point to the l2-ball just by divide each $v_i$ by a constant, but for l1-ball projection, you need to calculate the pendicular and the point on the l1-ball.



<figure>

  <img src="https://ws1.sinaimg.cn/large/006tNc79ly1fz83snhxsyj316f0u0k1w.jpg" width="250px"/>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/2D-simplex.svg/300px-2D-simplex.svg.png" width="100px"/>
  <figcaption>Fig.1 L1-Ball projection, l2-ball projection    Fig.2 3-D simplex example </figcaption>

</figure>


So the main contribution of the paper is `the derivation of gradient projections with l1 domain constraints that can be performed almost as fast as l2.`

Step 1: Prove the Euclidean proejction onto the simplex can be done in $O(nlogn)$.

Step 2: Because of the symmetry of l1-ball, the same solution above can be applied on l1-ball by first remove the sign, calculate the projection on simplex and then restore the sign.

Step 3: Improve the solution above to linear time based on a modification of the randomized median finding algorithm.
