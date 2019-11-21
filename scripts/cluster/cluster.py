from sklearn.datasets.samples_generator import make_blobs

import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


def draw_base(data, labels_true, s=20, labelsize=18):
    colors = sns.color_palette(n_colors=len(set(labels_true)))
    fig, ax = prepare_plot()
    draw_clusters(data, labels_true, ax=ax, s=s, colors=colors)
    make_square(fig, ax, labelsize=labelsize)
    plt.savefig('base.svg', format='svg')

def draw_density(data, s=20, labelsize=18):
    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=0.07, min_samples=10).fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    cl = sns.color_palette(n_colors=len(labels))
    colors = [cl[0], cl[2], cl[1], cl[3]]

    fig, ax = prepare_plot()
    draw_clusters(data=data, labels=labels, ax=ax, s=s, colors=colors)

    # Draw circles
    for i, (x,y) in enumerate(data):
        if 0.67 <= y <= 0.7 and 0.56 < x <= 0.58:
            ax.add_patch(
                patches.Circle(
                    (x, y),  # (x,y)
                    0.08,  # radius
                    alpha=1, facecolor="green", edgecolor="red", linewidth=2, linestyle='solid', fill=False)
            )

        if 0.76 <= y <= 0.8 and 0.4 < x <= 0.7:
            ax.add_patch(
                patches.Circle(
                    (x, y),  # (x,y)
                    0.08,  # radius
                    alpha=1, facecolor="green", edgecolor="red", linewidth=2, linestyle='dashed', fill=False)
            )
        if y <= 0.2 and  x >= 0.75:
            ax.add_patch(
                patches.Circle(
                    (x, y),  # (x,y)
                    0.08,  # radius
                    alpha=1, facecolor="green", edgecolor="red", linewidth=2, linestyle='dotted', fill=False)
            )

    make_square(fig, ax, labelsize=labelsize)
    plt.savefig('density.svg', format='svg')


###################################### hierarchy

def draw_hierarchy(data, file_name='hierarchy', s=20, labelsize=18):
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette

    fig, ax = prepare_plot()

    Z = linkage(X, 'single', metric='euclidean')
    labels = fcluster(Z, t=0.1, criterion='distance')
    print(labels)

    #replace single values with -1, so they appear black in the graphic
    unique, counts =np.unique(labels, return_counts=True)
    for uni, cou in zip(unique, counts):
        if cou == 1:
            i, = np.where(labels == uni)
            labels[i] = -1

    # Farben in richtige Reihenfolge bringen
    cl = sns.color_palette(n_colors=len(labels))
    colors = [cl[0], cl[2], cl[1], cl[3]]

    draw_clusters(data, labels, ax=ax, s=s, colors=colors)
    make_square(fig, ax, labelsize=labelsize)
    plt.savefig('hierarchy.svg', format='svg')

    fig, ax = prepare_plot()

    set_link_color_palette(['tab:blue', 'tab:green', 'tab:orange'])
    dendrogram(Z, color_threshold=0.1, above_threshold_color='black', no_labels=True, ax=ax)
    plt.axhline(0.1, color='red', linestyle='dashed')

    # Set the label sizes 
    ax.tick_params(labelsize=labelsize)
    ax.set_ylabel('Euclidian distance', fontsize=labelsize)
    plt.tight_layout()

    # Make the plotted graphic a square
    x0, y0, dx, dy = ax.get_position().bounds
    maxd = max(dx, dy)
    width = 6 * maxd / dx
    height = 6 * maxd / dy
    fig.set_size_inches((width, height))

    plt.savefig('hierarchy_dendro.svg', format='svg')



#################### PARTITIONING #####################
def draw_partioning(data, s=20, labelsize=18):
    from sklearn.cluster import KMeans
    from scipy.spatial import Voronoi, voronoi_plot_2d

    for miter in [1, 2, 5]:
        fig, ax = prepare_plot()
        k = KMeans(n_clusters=3, n_init=1, random_state=10, init='random', max_iter=miter).fit(X)
        cluster_centers = k.cluster_centers_

        # Farben in richtige Reihenfolge bringen
        cl = sns.color_palette(n_colors=len(k.labels_))
        colors = [cl[1], cl[2], cl[0], cl[3]]

        draw_clusters(data, k.labels_, ax=ax, s=s, colors=colors)

        sns.scatterplot(cluster_centers[:, 0], cluster_centers[:, 1], color='red', marker='X', s=100, ax=ax)

        plt.xlim(0,1.4)
        a = Voronoi(cluster_centers)
        regions, vertices = voronoi_finite_polygons_2d(a, radius=2)

        all_points = []
        for r in regions:
            points = []
            for elem in r:
                points.append(vertices[elem])
            all_points.append(np.array(points))

        for p in all_points:
            sns.lineplot(p[:,0], p[:,1], color='black', ax=ax, sort=False)


        make_square(fig, ax, labelsize=labelsize)
        plt.savefig(f'partitioning_{miter}.svg', format='svg')

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all([v >= 0 for v in vertices]):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

##### DISTRIBUTION #####

def draw_distribution(data, s=20, labelsize=18):
    from sklearn.mixture import GaussianMixture
    for miter in [14, 200]:
        gmm = GaussianMixture(n_components=3, max_iter=miter, random_state=2, init_params='random').fit(data)
        labels = gmm.predict(data)
        probs = gmm.predict_proba(data)
        fig, ax = prepare_plot()
        plot_gmm(gmm, data, ax=ax, s=s)
        make_square(fig=fig, ax=ax, labelsize=labelsize)
        plt.savefig(f'distribution_{miter}.svg', format='svg')

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    from matplotlib.patches import Ellipse

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))

def plot_gmm(gmm, data, s, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    # Farben in richtige Reihenfolge bringen
    cl = sns.color_palette(n_colors=len(labels))
    colors = [cl[2], cl[1], cl[0], cl[3]]
    draw_clusters(data=data, labels=labels, ax=ax, s=s, colors=colors)

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w, color in zip(gmm.means_, gmm.covariances_, gmm.weights_, ['tab:green', 'tab:orange', 'tab:blue']):

        draw_ellipse(pos, covar, alpha=w * w_factor, ax=ax, color=color)

###

def prepare_plot():
    fig, ax = plt.subplots()
    return fig, ax


def make_square(fig,ax, labelsize=18):
    plt.ylim(0, 1)
    plt.xlim(0, 1)

    x0, y0, dx, dy = ax.get_position().bounds
    maxd = max(dx, dy)
    width = 6 * maxd / dx
    height = 6 * maxd / dy
    ax.tick_params(labelsize=labelsize)
    fig.set_size_inches((width, height))

def draw_clusters(data, labels, ax, s, colors):

    unique_labels = set(labels)
    print(unique_labels)
    # colors = sns.color_palette(n_colors=len(unique_labels))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            print('no label')
            # Black used for noise.
            col = 'black' #[0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = data[class_member_mask]
        sns.scatterplot(xy[:, 0], xy[:, 1], color=col, ax=ax, s=s)

# #############################################################################
# Generate sample data
centers = [[0.15, 0.85], [0.5, 0.3], [0.65, 0.65]]
X, labels_true = make_blobs(n_samples=100, centers=centers, cluster_std=[0.04, 0.15, 0.06], random_state=0)

lsize=18
point_size=80
draw_base(data=X, labels_true=labels_true, s=point_size, labelsize=lsize)
draw_hierarchy(data=X, s=point_size, labelsize=lsize)

draw_density(data=X, s=point_size, labelsize=lsize)

draw_distribution(data=X, s=point_size, labelsize=lsize)

draw_partioning(data=X, s=point_size, labelsize=lsize)
