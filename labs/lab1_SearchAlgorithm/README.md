# My Completion Process


## KinoDynamic
<div align="center">

![image](https://github.com/huskydoge/AI-3603-Theory-and-application-of-artificial-intelligence/assets/91367324/f93b07dc-b97e-4319-9c71-d489e4e901c7)

</div>

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

### Code

```python
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt

points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
                   [2, 0], [2, 1], [2, 2]])
# 计算指定点的泰森多边形
vor = Voronoi(points)
# 绘制泰森多边形二维图像
fig = voronoi_plot_2d(vor)
plt.show()
```
* https://www.cnblogs.com/ttweixiao-IT-program/p/14374270.html
* [带边框的voronoi图求解](https://blog.csdn.net/weixin_50098648/article/details/125473652)

### Reference
https://zhuanlan.zhihu.com/p/144815425
* [Hybrid A* Algorithm](https://zhuanlan.zhihu.com/p/593406203)
* [自动驾驶运动规划-Hybird A*算法](https://zhuanlan.zhihu.com/p/139489196)
