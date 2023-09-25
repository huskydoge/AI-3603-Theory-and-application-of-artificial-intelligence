# My Completion Process


## [Reed-Shepp Expansion](https://zhuanlan.zhihu.com/p/122544884)

## Voronoi ﬁeld

### [Voronoi Diagram](https://mathworld.wolfram.com/VoronoiDiagram.html)
The partitioning of a plane with n points into convex polygons such that each polygon contains exactly one generating point and every point in a given polygon is closer to its generating point than to any other. A Voronoi diagram is sometimes also known as a Dirichlet tessellation. The cells are called Dirichlet regions, Thiessen polytopes, or Voronoi polygons.


Assume the sequence of Seed Points is : $\{x_1, x_2, \ldots, x_n \mid x_i \in R^d, i=1,2, \ldots, n\}$, where $R^d$ represents that all Seed Points are coordinate points in the $d$ dimensional space. The $n$ seed nodes divide the $d$dimensional space into $n$ cells. The mathematical definition of each Cell is as follows:


$$
V_i=\{x \in R_d \mid \forall j \neq i, d\left(x, x_i\right) \leq d\left(x, x_j\right)\}
$$

### Formulation
$$
\begin{aligned}
\rho_V(x, y)= & \left(\frac{\alpha}{\alpha+d_{\mathcal{O}}(x, y)}\right)\left(\frac{d_{\mathcal{V}}(x, y)}{d_{\mathcal{O}}(x, y)+d_{\mathcal{V}}(x, y)}\right) 
 \frac{\left(d_{\mathcal{O}}-d_{\mathcal{O}}^{\max }\right)^2}{\left(d_{\mathcal{O}}^{\max }\right)^2}
\end{aligned}
$$


### Reference
https://zhuanlan.zhihu.com/p/144815425
