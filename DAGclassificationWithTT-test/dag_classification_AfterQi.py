import xml.etree.ElementTree as ET
import random
from collections import OrderedDict

import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

networkSpeed = 20 * 1024 * 1024
bestVMCapacity = 26 #26考虑最好机器，1不考虑

class SparseMat(object):
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape
        self.row = []
        self.col = []
        self.data = []

    def add(self, row, col, data):
        self.row.append(row)
        self.col.append(col)
        self.data.append(data)

    def get_col(self):
        return np.array(self.col)

    def get_row(self):
        return np.array(self.row)

    def get_data(self):
        return np.array(self.data)

def get_init_frontier(job_dag, depth):
    """
    Get the initial set of frontier nodes, based on the depth
    """
    sources = set(job_dag)
    for d in range(depth):
        new_sources = set()
        for n in sources:
            if len(n.child_nodes) == 0:
                new_sources.add(n)
            else:
                new_sources.update(n.child_nodes)
        sources = new_sources
    frontier = sources
    return frontier

def absorb_sp_mats(in_mats, depth):
    """
    Merge multiple sparse matrices to
    a giant one on its diagonal
    e.g.,
    [0, 1, 0]    [0, 1, 0]    [0, 0, 1]
    [1, 0, 0]    [0, 0, 1]    [0, 1, 0]
    [0, 0, 1]    [1, 0, 0]    [0, 1, 0]
    to
    [0, 1, 0]
    [1, 0, 0]   ..  ..    ..  ..
    [0, 0, 1]
              [0, 1, 0]
     ..  ..   [0, 0, 1]   ..  ..
              [1, 0, 0]
                        [0, 0, 1]
     ..  ..    ..  ..   [0, 1, 0]
                        [0, 1, 0]
    where ".." are all zeros
    depth is on the 3rd dimension,
    which is orthogonal to the planar
    operations above
    output SparseTensorValue from tensorflow
    """
    sp_mats = []
    for d in range(depth):
        row_idx = []
        col_idx = []
        data = []
        shape = 0
        base = 0
        for m in in_mats:
            row_idx.append(m[d].get_row() + base)
            col_idx.append(m[d].get_col() + base)
            data.append(m[d].get_data())
            shape += m[d].shape[0]
            base += m[d].shape[0]
        row_idx = np.hstack(row_idx)
        col_idx = np.hstack(col_idx)
        data = np.hstack(data)
        indices = np.mat([row_idx, col_idx]).transpose()
        sp_mats.append(tf.SparseTensorValue(
            indices, data, (shape, shape)))
    return sp_mats


def merge_masks(masks, MaxDepth):
    """
    e.g.,
    [0, 1, 0]  [0, 1]  [0, 0, 0, 1]
    [0, 0, 1]  [1, 0]  [1, 0, 0, 0]
    [1, 0, 0]  [0, 0]  [0, 1, 1, 0]
    to
    a list of
    [0, 1, 0, 0, 1, 0, 0, 0, 1]^T,
    [0, 0, 1, 1, 0, 1, 0, 0, 0]^T,
    [1, 0, 0, 0, 0, 0, 1, 1, 0]^T
    Note: mask dimension d is pre-determined
    """
    merged_masks = []
    for d in range(MaxDepth):
        merged_mask = []
        for mask in masks:
            merged_mask.append(mask[d:d+1, :].transpose())
        if len(merged_mask) > 0:
            merged_mask = np.vstack(merged_mask)
        merged_masks.append(merged_mask)
    return merged_masks

def get_bottom_up_paths(job_dag, MaxDepth):
    """
    The paths start from all leaves and end with
    frontier (parents all finished) unfinished nodes
    """
    num_nodes = len(job_dag)
    # 将图根据上下依赖分成了MaxDepth层，列表msg_mat存放了各层的邻接矩阵，
    # smsg_mats[d]存放了d层的邻接矩阵，矩阵维度为taskNum*taskNum
    msg_mats = []
    # 将图根据上下依赖分成了MaxDepth层，列表msg_masks存放了各层的父任务的位置，
    # smsg_mats[d]存放了d层的父任务的位置，存放形式为列向量taskNum*1，d层父任务的index置为1
    msg_masks = np.zeros([MaxDepth, num_nodes])
    # 将图根据上下依赖分成了MaxDepth层，列表trans_mats存放了各层的传输时间矩阵，
    # trans_mats[d]存放了d层的传输时间矩阵，矩阵维度为taskNum*taskNum
    trans_mats = []
    # get set of frontier nodes in the beginning
    # this is constrained by the message passing depth
    frontier = get_init_frontier(job_dag, MaxDepth)
    msg_level = {}
    # initial nodes are all message passed
    for n in frontier:
        msg_level[n] = 0
    # pass messages
    for depth in range(MaxDepth):
        new_frontier = set()
        parent_visited = set()  # save some computation
        for n in frontier:
            for parent in n.parent_nodes:
                if parent not in parent_visited:
                    curr_level = 0
                    children_all_in_frontier = True
                    for child in parent.child_nodes:
                        if child not in frontier:
                            children_all_in_frontier = False
                            break
                        if msg_level[child] > curr_level:
                            curr_level = msg_level[child]
                    # children all ready
                    if children_all_in_frontier:
                        if parent not in msg_level or \
                           curr_level + 1 > msg_level[parent]:
                            # parent node has deeper message passed
                            new_frontier.add(parent)
                            msg_level[parent] = curr_level + 1
                    # mark parent as visited
                    parent_visited.add(parent)
        if len(new_frontier) == 0:
            break  # some graph is shallow
        # assign parent-child path in current iteration
        sp_mat = SparseMat(dtype=np.float32, shape=(num_nodes, num_nodes))
        sp_mat1 = SparseMat(dtype=np.float32, shape=(num_nodes, num_nodes))
        for n in new_frontier:
            for child in n.child_nodes:
                sp_mat.add(row=n.idx, col=child.idx, data=1) #父任务与子任务的依赖矩阵
                sp_mat1.add(row=n.idx, col=child.idx, data=calculateTransferTime(n, child)) #父任务与子任务的传输时间矩阵
            msg_masks[depth, n.idx] = 1
        msg_mats.append(sp_mat)
        trans_mats.append(sp_mat1)
        # Note: there might be residual nodes that
        # can directly pass message to its parents
        # it needs two message passing steps
        # (e.g., TPCH-17, node 0, 2, 4)
        for n in frontier:
            parents_all_in_frontier = True
            for p in n.parent_nodes:
                if not p in msg_level:
                    parents_all_in_frontier = False
                    break
            if not parents_all_in_frontier:
                new_frontier.add(n)
        # start from new frontier
        frontier = new_frontier
    # deliberately make dimension the same, for batch processing
    for _ in range(depth, MaxDepth):
        msg_mats.append(SparseMat(dtype=np.float32,
            shape=(num_nodes, num_nodes)))
        trans_mats.append(SparseMat(dtype=np.float32,
                                  shape=(num_nodes, num_nodes)))
    msg_mats = absorb_sp_mats([msg_mats], MaxDepth)
    trans_mats = absorb_sp_mats([trans_mats], MaxDepth)
    msg_masks = merge_masks([msg_masks], MaxDepth)
    trans_merge = np.ones((num_nodes, 1))
    # print("trans_merge: ", trans_merge, "type", type(trans_merge))
    # print("trans_merge: ", trans_merge.shape)
    return msg_mats, msg_masks, trans_mats, trans_merge

class FileItem(object):
    def __init__(self, name, size, type):
        self.name = name
        self.size = float(size)
        if self.size < 0:
            self.size = - self.size
        self.type = type #文件类型：input，output

class DAGNode(object):
    def __init__(self, id, data, idx):
        self.id = id
        self.attrib = float(data)
        if self.attrib < 0:
            self.attrib = - self.attrib
        elif self.attrib == 0:
            self.attrib = 0.0000001
        self.parent_nodes = []
        self.child_nodes = []
        self.file_list = []
        self.idx = idx

    def add_parent(self, node):
        self.parent_nodes.append(node)

    def add_child(self, node):
        self.child_nodes.append(node)

    def add_file(self, file):
        self.file_list.append(file)

def read_dag_from_xml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    DAG_Nodes = []
    DAG_attrib = []
    idx = 0
    for elem in root:
        job = elem.attrib
        if 'namespace' in job.keys():
            node_ID = job['id']
            node_attrib = float(job['runtime']) / bestVMCapacity
            new_node = DAGNode(node_ID, node_attrib, idx)
            #读该node的文件
            for subelem in elem:
                uses = subelem.attrib
                tfile = FileItem(uses['file'], uses['size'], uses['link'])
                new_node.add_file(tfile)
            DAG_Nodes.append(new_node)
            idx = idx + 1
        else:
            node_ID = job['ref']
            for node_ele_1 in DAG_Nodes:
                if node_ele_1.id == node_ID:
                    for subelem in elem:
                        parents = subelem.attrib
                        parents_id = parents['ref']
                        for node_ele_2 in DAG_Nodes:
                            if node_ele_2.id == parents_id:
                                node_ele_1.add_parent(node_ele_2)
                                node_ele_2.add_child(node_ele_1)
    for node_ele in DAG_Nodes:
        DAG_attrib.append(node_ele.attrib)
    DAG_attrib_a = np.array(DAG_attrib)
    DAG_attrib_a = DAG_attrib_a.reshape(-1, 1)
    return DAG_Nodes, DAG_attrib_a

def calculateTransferTime(parent, child):
    acc = 0.0
    for parentFile in parent.file_list:
        if parentFile.type != 'output':
            continue
        for childFile in child.file_list:
            if childFile.type == 'input' and childFile.name == parentFile.name:
                acc += float(childFile.size)/networkSpeed
                break
    return acc

def zeros(shape, dtype=tf.float32, scope='default'):
    with tf.variable_scope(scope):
        init = tf.zeros(shape, dtype=dtype)
        return tf.Variable(init)

def glorot(shape, dtype=tf.float32, scope='default'):
    with tf.variable_scope(scope):
        init = tf.random_uniform(
            shape, minval=-1.0, maxval=1.0, dtype=dtype)
        return tf.Variable(init)

"""
Graph Convolutional Network
Propergate node features among neighbors
via parameterized message passing scheme
"""
class GraphCNN(object):
    def __init__(self, inputs, input_dim, hid_dims, output_dim,
                 max_depth, act_fn, scope='gcn'):
        self.inputs = inputs
        self.input_dim = input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.act_fn = act_fn
        self.scope = scope
        # message passing
        self.adj_mats = [tf.sparse_placeholder(
            tf.float32, [None, None]) for _ in range(self.max_depth)]
        self.masks = [tf.placeholder(
            tf.float32, [None, 1]) for _ in range(self.max_depth)]
        self.trans_mats = [tf.sparse_placeholder(
            tf.float32, [None, None]) for _ in range(self.max_depth)]
        self.trans_merge = tf.placeholder(tf.float32, [None, 1]) #n*1的列向量，值全为1
        # initialize message passing transformation parameters
        # h: x -> x'
        self.prep_weights, self.prep_bias = \
            self.init(self.input_dim, self.hid_dims, self.output_dim)
        # f: x' -> e
        self.proc_weights, self.proc_bias = \
            self.init(self.output_dim, self.hid_dims, self.output_dim)
        # g: e -> e
        self.agg_weights, self.agg_bias = \
            self.init(self.output_dim + 1, self.hid_dims, self.output_dim) #不考虑传世时间只考虑依赖，去掉+1 self.init(self.output_dim + 1, self.hid_dims, self.output_dim)
        # graph message passing, self.outputs是self.inputs聚合邻居信息后的值
        self.outputs = self.forward()

    def init(self, input_dim, hid_dims, output_dim):
        # Initialize the parameters
        # these weights may need to be re-used
        # e.g., we may want to propagate information multiple times
        # but using the same way of processing the nodes
        weights = []
        bias = []
        curr_in_dim = input_dim
        # hidden layers
        for hid_dim in hid_dims:
            weights.append(
                glorot([curr_in_dim, hid_dim], scope=self.scope))
            bias.append(
                zeros([hid_dim], scope=self.scope))
            curr_in_dim = hid_dim
        # output layer
        weights.append(glorot([curr_in_dim, output_dim], scope=self.scope))
        bias.append(zeros([output_dim], scope=self.scope))
        return weights, bias

    def forward(self):
        # message passing among nodes
        # the information is flowing from leaves to roots
        x = self.inputs
        # raise x into higher dimension
        for l in range(len(self.prep_weights)):
            x = tf.matmul(x, self.prep_weights[l])
            x += self.prep_bias[l]
            x = self.act_fn(x)
        for d in range(self.max_depth):
            # work flow: index_select -> f -> masked assemble via adj_mat -> g
            y = x
            # process the features on the nodes
            for l in range(len(self.proc_weights)):
                y = tf.matmul(y, self.proc_weights[l])
                y += self.proc_bias[l]
                y = self.act_fn(y)
            # message passing
            y = tf.sparse_tensor_dense_matmul(self.adj_mats[d], y)
            trans = tf.sparse_tensor_dense_matmul(self.trans_mats[d], self.trans_merge) #n*n矩阵乘n*1得到n*1
            # aggregate child features
            for l in range(len(self.agg_weights)):
                if l < 1: #if到else是为了传输时间加的，去掉就是不考虑传输时间，只考虑依赖的
                    y = tf.matmul(tf.concat([y, trans], axis=1), self.agg_weights[l])
                    y += self.agg_bias[l]
                    y = self.act_fn(y)
                else:
                    y = tf.matmul(y, self.agg_weights[l])
                    y += self.agg_bias[l]
                    y = self.act_fn(y)
            # remove the artifact from the bias term in g
            y = y * self.masks[d]
            # assemble neighboring information
            x = x + y
        return x

def get_unfinished_nodes_summ_mat(job_dag):
    total_num_nodes = np.sum([len(job_dag)])
    summ_row_idx = []
    summ_col_idx = []
    summ_data = []
    summ_shape = (1, total_num_nodes)
    base = 0
    j_idx = 0
    for node in job_dag:
        summ_row_idx.append(j_idx)
        summ_col_idx.append(base + node.idx)
        summ_data.append(1)
    base += len(job_dag)
    j_idx += 1
    summ_indices = np.mat([summ_row_idx, summ_col_idx]).transpose()
    summerize_mat = tf.SparseTensorValue(summ_indices, summ_data, summ_shape)
    return summerize_mat

"""
Graph Summarization Network
Summarize node features globally
via parameterized aggregation scheme
"""
class GraphSNN(object):
    def __init__(self, inputs, input_dim, hid_dims, output_dim, act_fn, scope='gsn'):
        # on each transformation, input_dim -> (multiple) hid_dims -> output_dim
        self.inputs = inputs
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dims = hid_dims
        self.act_fn = act_fn
        self.scope = scope
        # graph summarization, hierarchical structure
        self.summ_mats = tf.sparse_placeholder(tf.float32, [None, None])
        # initialize summarization parameters for each hierarchy
        self.dag_weights, self.dag_bias = \
            self.init(self.input_dim, self.hid_dims, self.output_dim)
        # graph summarization operation
        self.summaries = self.summarize()

    def init(self, input_dim, hid_dims, output_dim):
        # Initialize the parameters
        # these weights may need to be re-used
        # e.g., we may want to propagate information multiple times
        # but using the same way of processing the nodes
        weights = []
        bias = []
        curr_in_dim = input_dim
        # hidden layers
        for hid_dim in hid_dims:
            weights.append(
                glorot([curr_in_dim, hid_dim], scope=self.scope))
            bias.append(
                zeros([hid_dim], scope=self.scope))
            curr_in_dim = hid_dim
        # output layer
        weights.append(glorot([curr_in_dim, output_dim], scope=self.scope))
        bias.append(zeros([output_dim], scope=self.scope))
        return weights, bias

    def summarize(self):
        # DAG level summary
        s = self.inputs
        for i in range(len(self.dag_weights)):
            s = tf.matmul(s, self.dag_weights[i])
            s += self.dag_bias[i]
            s = self.act_fn(s)
        ret = tf.sparse_tensor_dense_matmul(self.summ_mats, s)
        return ret

class FC_classify(object):
    def __init__(self, inputs, input_dim, hid_dims, class_num, act_fn, scope='fcc'):
        self.inputs = inputs
        self.input_dim = input_dim
        self.hid_dims = hid_dims
        self.output_dim = class_num
        self.act_fn = act_fn
        self.scope = scope
        self.dag_weights, self.dag_bias = self.init(self.input_dim, self.hid_dims, self.output_dim)
        self.result = self.classification()

    def init(self, input_dim, hid_dims, output_dim):
        weights = []
        bias = []
        curr_in_dim = input_dim
        # hidden layers
        for hid_dim in hid_dims:
            weights.append(
                glorot([curr_in_dim, hid_dim], scope=self.scope))
            bias.append(
                zeros([hid_dim], scope=self.scope))
            curr_in_dim = hid_dim
        # output layer
        weights.append(glorot([curr_in_dim, output_dim], scope=self.scope))
        bias.append(zeros([output_dim], scope=self.scope))
        return weights, bias

    def classification(self):
        s = self.inputs
        for i in range(len(self.dag_weights)):
            s = tf.matmul(s, self.dag_weights[i])
            s += self.dag_bias[i]
            s = self.act_fn(s)
        return s

def main():
    node_input_dim = 1
    hid_dims = [4]
    output_dim = 4
    class_num = 4
    max_depth = 10
    act_fn = tf.nn.leaky_relu
    act_fn_0 = tf.nn.softmax #齐天宇试过这个激活函数
    deadline_num = 10
    workflow2Index = []
    classificationResult = OrderedDict()

    '''
        读取样本
    '''
    # dir_path = "E:/0Work/1Ideas/workflowClassificationAndScheduling/workflow2subDLable/SyntheticWorkflows"
    dir_path = "E:/0Work/1Ideas/workflowClassificationAndScheduling/workflowScheduling/workflowGenerator/WorkflowGenerator/bharathi/generatedWorkflows"
    epoch = 200
    '''
    for file in os.listdir(dir_path):
        if file.endswith(".dax"):
            name_string = file.split(".")
            if int(name_string[2]) <= 1000:
                filename = os.path.join(dir_path, file)
                ag_info, dag_data = read_dag_from_xml(filename)
    '''
    dag_info_list = []
    dag_data_list = []
    # folder_list = ["CYBERSHAKE","GENOME","LIGO", "MONTAGE", "SIPHT"]
    # name_num = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    folder_list = ["CyberShake","Epigenomics","Inspiral", "Montage", "Sipht"]
    folder_num = 0 #100个文件时用的: =0是C开始，=1是E开始，=2是LIGO开始，……
    name_num = [30, 50, 100, 1000]
    # folder_list = ["CYBERSHAKE"]
    # name_num = [50]
    for k in range(len(folder_list)):
            for i in range(len(name_num)):
                for j in range(200): #100, 200, 400
                    # file_name = dir_path + "/" +folder_list[k] + "/"+folder_list[k]+".n."+str(name_num[i])+"."+str(j)+".dax"
                    file_name = dir_path + "/" + folder_list[k] + "_" + str(
                        name_num[i]) + "." + str(j) + ".xml"
                    dag_info, dag_data = read_dag_from_xml(file_name)
                    dag_info_list.append(dag_info)
                    dag_data_list.append(dag_data)
    '''
    用HEFT_paper3.dax作为例子，调试学习
    '''
    # file_name = "E:/0Work/1Ideas/workflowClassificationAndScheduling/workflow2subDLable/SyntheticWorkflows/Sample
    # /HEFT_paper3.dax" dag_info, dag_data = read_dag_from_xml(file_name) dag_info_list.append(dag_info)
    # dag_data_list.append(dag_data)

    '''
        读取样本标签
    '''
    label_list = []
    with open('E:/0Work/1Ideas/workflowClassificationAndScheduling/workflowScheduling/workflow2subDLable/result/repeat'
              '/workflow2label2_200.txt') as f: # workflow2label4， workflow2label2_200， workflow2label2_400 /repeat
        lines = f.readlines()
    for line in lines:
        label_info = line.split(", ")
        #label = tf.one_hot([int(label_info[1])], class_num)
        label = np.zeros((1, class_num))
        label[0][int(label_info[2])-1] = 1
        #label[0] = label[0] * 0.9 + 0.025 #齐天宇 label smooth 无用
        label_list.append(label)
        workflow2Index.append(label_info[0]+', '+label_info[1])
        classificationResult.update({label_info[0]+', '+label_info[1]: -1})

    '''
    读取样本deadlines
    '''
    deadline_list = []
    with open('E:/0Work/1Ideas/workflowClassificationAndScheduling/workflowScheduling/workflow2subDLable/result/repeat'
              '/deadlineFactors2_200.txt') as f:    # deadlines2, deadlineFactors2, deadlines2_200, deadlineFactors2_200, deadlineFactors2_400 /repeat
        lines = f.readlines()
    for line in lines:
        deadline_list.append(float(line))

    '''
        训练集和测试集的划分
    '''
    training_set = []
    testing_set = []
    '''原先的'''
    training_rate = 13
    count_n = 0
    while count_n < len(dag_info_list):
        if count_n % training_rate == 0:
            testing_set.append(count_n)
        else:
            training_set.append(count_n)
        count_n = count_n + 1

    '''齐天宇的：对数据集的划分进⾏了优化，原始划分数据集并不属于随机划分，容易造成最终学习效果位于平
        均正确率之下。故将数据集改为了随机分布，并增⼤⼀些训练集⽐例'''
    # label_shuf = []
    # label_sing = []
    # workflow2Index_shuf = []
    # workflow2Index_sing = []
    # deadline_shuf = []
    # deadline_sing = []
    # if (len(label_list) % 10 == 0):
    #     for idx in range(0, len(label_list), 10):
    #         for i in range(10):
    #             label_sing.append(label_list[idx + i])
    #             workflow2Index_sing.append(workflow2Index[idx + i])
    #             deadline_sing.append(deadline_list[idx + i])
    #         label_shuf.append(label_sing)
    #         workflow2Index_shuf.append(workflow2Index_sing)
    #         deadline_shuf.append(deadline_sing)
    #         label_sing = []
    #         workflow2Index_sing = []
    #         deadline_sing = []
    # else:
    #     print("IndexError: list index out of range.")
    # # print(len(label_shuf))
    # # print(len(workflow2Index_shuf))
    # # print(len(deadline_shuf))
    #
    # c = list(zip(label_shuf, workflow2Index_shuf, deadline_shuf, dag_info_list, dag_data_list))
    # random.shuffle(c)
    # label_shuf, workflow2Index_shuf, deadline_shuf, dag_info_list, dag_data_list = zip(*c)
    # label_list = [n for a in label_shuf for n in a]
    # workflow2Index = [n for a in workflow2Index_shuf for n in a]
    # deadline_list = [n for a in deadline_shuf for n in a]
    #
    # count = []
    # for i in range(len(dag_info_list)):
    #     count.append(i)
    # training_set, testing_set = train_test_split(count, train_size=0.90, shuffle=False)


    '''
    用HEFT_paper3.dax作为例子，调试学习
    '''
    # testing_set.append(count_n)
    # training_set.append(count_n)

    g = tf.Graph()
    with g.as_default():
        # node input dimension: [total_num_nodes, num_features]
        node_inputs = tf.placeholder(tf.float32, [None,  node_input_dim])
        label_tf = tf.placeholder(tf.float32, [1,  class_num])
        deadline_tf = tf.placeholder(tf.float32, [1,  1])
        node_size_tf = tf.placeholder(tf.float32, [1,  1])
        gcn = GraphCNN(node_inputs, node_input_dim, hid_dims, output_dim, max_depth, act_fn)
        gsn = GraphSNN(tf.concat([node_inputs, gcn.outputs], axis=1), node_input_dim + output_dim, \
                                                                 hid_dims, output_dim, act_fn)
        #原先的
        # graph_class = FC_classify(gsn.summaries, output_dim, [], class_num, act_fn)
        #在FC层的输入中加了deadline
        graph_class = FC_classify(tf.concat([gsn.summaries, deadline_tf], axis=1), 1 + output_dim, [4,4], class_num, act_fn)
        # 在FC层的输入中加了工作流中的任务个数
        # graph_class = FC_classify(tf.concat([gsn.summaries, node_size_tf], axis=1), 1 + output_dim, [], class_num,
        #                           act_fn)
        #在FC层的输入中加了deadline和工作流中的任务个数 [5, 5]
        # graph_class = FC_classify(tf.concat([gsn.summaries, deadline_tf, node_size_tf], axis=1), 2 + output_dim, [5], class_num,
        #                           act_fn)
        loss_op = tf.nn.softmax_cross_entropy_with_logits(labels=label_tf, logits=graph_class.result)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.003).minimize(loss_op)
        # optimizer = tf.train.AdamOptimizer(learning_rate=0.0015).minimize(loss_op) #齐天宇
        with tf.Session(graph=g) as sess:
            sess.run(tf.global_variables_initializer())
            accuracy_curve = []
            loss_curve = []
            test_curve = []
            test_loss_curve = []
            for i in range(epoch):
                right_count = 0
                epoch_loss = 0
                test_epoch_loss = 0
                sample_num = len(training_set)
                sample_index_list = list(range(sample_num))
                random.shuffle(sample_index_list)
                for j in range(sample_num):
                    sample_index = training_set[sample_index_list[j]]
                    gcn_mats, gcn_masks, gcnTran_mats, gcnTrans_merge = get_bottom_up_paths(dag_info_list[sample_index], max_depth)
                    summ_mats = get_unfinished_nodes_summ_mat(dag_info_list[sample_index])
                    dag_data = dag_data_list[sample_index]
                    node_size = len(dag_data)
                    node_size = np.array(node_size)
                    node_size = node_size.reshape(-1, 1)
                    for dead in range(deadline_num):
                        # true_sample_index = sample_index*deadline_num + dead + 4000*folder_num #100个文件时用的: =0是C开始，=1是E开始，=2是LIGO开始，
                        true_sample_index = sample_index * deadline_num + dead
                        label = label_list[true_sample_index]
                        deadline = deadline_list[true_sample_index]
                        deadline = np.array(deadline)
                        deadline = deadline.reshape(-1, 1)
                        # _, loss, graph_ret, graph_vector = sess.run([optimizer, loss_op, graph_class.result, gsn.summaries], \
                        #                         feed_dict={i: d for i, d in zip( \
                        #                         [node_inputs] + gcn.adj_mats + gcn.masks + [gsn.summ_mats] + [label_tf], \
                        #                         [dag_data] + gcn_mats + gcn_masks + [summ_mats] + [label])})
                        _,loss,graph_ret,graph_vector = sess.run([optimizer, loss_op, graph_class.result, gsn.summaries], \
                                                feed_dict={i: d for i, d in zip( \
                                                 [node_inputs] + gcn.adj_mats + gcn.masks + gcn.trans_mats + [gcn.trans_merge] + [gsn.summ_mats] + [label_tf] + [deadline_tf], \
                                                 [dag_data] + gcn_mats + gcn_masks + gcnTran_mats + [gcnTrans_merge] + [summ_mats] + [label] + [deadline])})
                        # _,loss,graph_ret,graph_vector = sess.run([optimizer, loss_op, graph_class.result, gsn.summaries], \
                        #                         feed_dict={i: d for i, d in zip( \
                        #                          [node_inputs] + gcn.adj_mats + gcn.masks + [gsn.summ_mats] + [label_tf] + [node_size_tf], \
                        #                          [dag_data] + gcn_mats + gcn_masks + [summ_mats] + [label] + [node_size])})
                        # _,loss,graph_ret,graph_vector = sess.run([optimizer, loss_op, graph_class.result, gsn.summaries], \
                        #                         feed_dict={i: d for i, d in zip( \
                        #                          [node_inputs] + gcn.adj_mats + gcn.masks + [gsn.summ_mats] + [label_tf] + [deadline_tf] + [node_size_tf], \
                        #                          [dag_data] + gcn_mats + gcn_masks + [summ_mats] + [label] + [deadline] + [node_size])})

                        #print(graph_vector)
                        #print(graph_ret)
                        #print(loss)
                        #print(i)
                        epoch_loss = loss[0] + epoch_loss
                        if np.argmax(graph_ret) == np.argmax(label):
                            right_count = right_count + 1
                        #存储分类结果
                        classificationResult.update({workflow2Index[true_sample_index]: np.argmax(graph_ret)+1})
                        '''
                        for z in range(len(gcn.prep_weights)):
                            print("gcn_prep_weights: ", sess.run(gcn.prep_weights[z]))
                        for z in range(len(gcn.proc_weights)):
                            print("gcn_proc_weights: ", sess.run(gcn.proc_weights[z]))
                        for z in range(len(gcn.agg_weights)):
                            print("gcn_agg_weights: ", sess.run(gcn.agg_weights[z]))
                        for z in range(len(gsn.dag_weights)):
                            print("gsn_weights: ", sess.run(gsn.dag_weights[z]))
                        for z in range(len(graph_class.dag_weights)):
                            print("fc_weights: ", sess.run(graph_class.dag_weights[z]))
                        '''

                accuracy_rate = right_count/(sample_num*deadline_num)
                print('epoch: ', i)
                print("the correcting rate is: ", accuracy_rate)
                accuracy_curve.append(accuracy_rate)
                average_loss = epoch_loss/(sample_num*deadline_num)
                print("the epoch average loss is: ", average_loss)
                loss_curve.append(average_loss)

                bingo_num = 0
                for j in range(len(testing_set)):
                    sample_index = testing_set[j]
                    gcn_mats, gcn_masks, gcnTran_mats, gcnTrans_merge = get_bottom_up_paths(dag_info_list[sample_index], max_depth)
                    summ_mats = get_unfinished_nodes_summ_mat(dag_info_list[sample_index])
                    dag_data = dag_data_list[sample_index]
                    node_size = len(dag_data)
                    node_size = np.array(node_size)
                    node_size = node_size.reshape(-1, 1)
                    for dead in range(deadline_num):
                        true_sample_index = sample_index*deadline_num + dead+4000*folder_num
                        label = label_list[true_sample_index]
                        deadline = deadline_list[true_sample_index]
                        deadline = np.array(deadline)
                        deadline = deadline.reshape(-1, 1)
                        # ret = sess.run([graph_class.result], feed_dict={i: d for i, d in zip([node_inputs] + gcn.adj_mats + \
                        #                      gcn.masks + [gsn.summ_mats], [dag_data] + gcn_mats + gcn_masks + [summ_mats])})
                        ret = sess.run([graph_class.result], feed_dict={i: d for i, d in zip(
                            [node_inputs] + gcn.adj_mats + gcn.masks + gcn.trans_mats + [gcn.trans_merge] + [gsn.summ_mats] + [deadline_tf],
                            [dag_data] + gcn_mats + gcn_masks + gcnTran_mats + [gcnTrans_merge] + [summ_mats] + [deadline])})
                        # ret = sess.run([graph_class.result], feed_dict={i: d for i, d in zip([node_inputs] + gcn.adj_mats + \
                        #                      gcn.masks + [gsn.summ_mats] + [node_size_tf], [dag_data] + gcn_mats + gcn_masks + [summ_mats] + [node_size])})
                        # loss,ret = sess.run([loss_op, graph_class.result], feed_dict={i: d for i, d in zip(
                        #     [node_inputs] + gcn.adj_mats + gcn.masks + [gsn.summ_mats] + [label_tf] + [deadline_tf] + [node_size_tf],
                        #     [dag_data] + gcn_mats + gcn_masks + [summ_mats] + [label] + [deadline] + [node_size])})

                        class_ret = np.argmax(ret)
                        label = np.argmax(label_list[true_sample_index])
                        if label == class_ret:
                            bingo_num = bingo_num + 1
                        test_epoch_loss = loss[0] + test_epoch_loss
                        # 存储分类结果
                        classificationResult.update({workflow2Index[true_sample_index]: class_ret + 1})
                correct_rate = bingo_num/(len(testing_set)*deadline_num)
                test_curve.append(correct_rate)
                average_loss = test_epoch_loss/(len(testing_set)*deadline_num)
                test_loss_curve.append(average_loss)
                print("testing result in this epoch is: ", correct_rate)
                print("the epoch average testing loss is: ", average_loss)

                file_handle = open('./result/epoch' + str(i) + '_classificationResult.txt', mode='w+')
                for key in classificationResult.keys():
                    file_handle.write(key + ', ' + str(classificationResult[key]) + '\n')
                file_handle.close()

            # f = open("out.txt", "w")
            # print("the accuracy curve is: ", accuracy_curve, file=f)
            print("the accuracy curve is: ", accuracy_curve)
            print("the average loss curve is: ", loss_curve)
            print("the testing accuracy is: ", test_curve)
            print("the average test loss curve is: ", test_loss_curve)

            # 训练过程可视化
            fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
            fig.suptitle('Training Metrics test')

            axes[0].set_ylabel("Loss", fontsize=14)
            axes[0].plot(loss_curve, 'r')
            axes[0].plot(test_loss_curve, 'b')

            axes[1].set_ylabel("Accuracy", fontsize=14)
            axes[1].plot(accuracy_curve, 'r')
            axes[1].plot(test_curve, 'b')

            plt.show()

if __name__ == '__main__':
    main()
