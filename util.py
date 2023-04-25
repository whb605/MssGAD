import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def node_iter(G):
    # if float(nx.__version__)<2.0:
    #     return G.nodes()
    # else:
    #     return G.nodes
    return G.nodes


def node_dict(G):
    # if float(nx.__version__)>2.1:
    #     node_dict = G.nodes
    # else:
    #     node_dict = G.node
    # return node_dict
    return G.nodes


def cal_score(x, n_clusters):
    if x.ndim == 1:
        x = x.reshape(len(x), -1 )
    elif x.ndim == 2:
        pass
    else:
        print('Dimension Must be 1D or 2D')
    kclf = KMeans(n_clusters = n_clusters)
    kclf.fit(x)
    score = silhouette_score(x, kclf.labels_, metric='euclidean')
    return score


def mk_dir(path):
    path = path.strip().rstrip('\\')
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    else:
        return False


def text_create(path ,name, msg):
    desktop_path = path  # 新创建的txt文件的存放路径
    full_path = desktop_path + name + '.txt'  # 也可以创建一个.doc的word文档
    file = open(full_path, 'w')
    file.write(msg)  # msg也就是下面的Hello world!
    file.close()

