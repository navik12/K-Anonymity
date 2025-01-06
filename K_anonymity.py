import csv
import glob
import itertools
import os
import numpy as np


def file_costLM(raw_dataset, anonymized_dataset, DGH_folder) -> float:
    assert (len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
            and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    att_list = list(DGHs.keys())
    firsts = DGH_firstele(DGH_folder)
    cost = 0
    quasi_atts = []

    person_att_list = list(raw_dataset[0].keys())
    for j in person_att_list:
        if j in att_list:
            quasi_atts.append(j)
    weight = len(quasi_atts)
    for i in range(len(raw_dataset)):
        for j in quasi_atts:
            anon_att_name = anonymized_dataset[i][j]
            anon_LM = (find_leaf_number(DGHs[j][anon_att_name]) - 1) / (find_leaf_number(firsts[j]) - 1)
            cost += anon_LM * 1 / weight
    return cost

def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
            DGH_folder: str) -> float:
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert (len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
            and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    att_list = list(DGHs.keys())
    cost = 0
    for i in range(len(raw_dataset)):
        # e.g: ['age', 'workclass', 'education', 'income' ... ]
        person_att_list = list(raw_dataset[i].keys())
        for j in person_att_list:
            if j in att_list:
                # e.g: 17
                raw_att_name = raw_dataset[i][j]
                # e.g: [10,20)
                anon_att_name = anonymized_dataset[i][j]
                raw_level = DGHs[j][raw_att_name].level
                anon_level = DGHs[j][anon_att_name].level
                cost += abs(raw_level - anon_level)

    return cost


def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
            DGH_folder: str) -> float:
    raw_dataset = read_dataset(raw_dataset_file)  # returns list of dicts
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert (len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
            and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    att_list = list(DGHs.keys())
    firsts = DGH_firstele(DGH_folder)
    cost = 0
    quasi_atts = []
    # e.g: ['age', 'workclass', 'education', 'income' ... ]
    person_att_list = list(raw_dataset[0].keys())
    for j in person_att_list:
        if j in att_list:
            quasi_atts.append(j)
    weight = len(quasi_atts)
    for i in range(len(raw_dataset)):
        for j in quasi_atts:
            anon_att_name = anonymized_dataset[i][j]
            anon_LM = (find_leaf_number(DGHs[j][anon_att_name]) - 1) / (find_leaf_number(firsts[j]) - 1)
            cost += anon_LM * 1 / weight
    return cost

class Node(object):
    def __init__(self, name: str, level: int, parent=None):
        self.name = name
        self.level = level
        self.children = []
        self.parent = parent

    def __str__(self):
        return "Node: " + self.name

    def add_child(self, obj):
        self.children.append(obj)

def find_leaf_number(node: Node) -> int:
    ret = 0
    # if children is empty, the node is a leaf!
    if len(node.children) == 0:
        return 1
    else:
        for child in node.children:
            ret += find_leaf_number(child)
        return ret

def low_com_ans_lis(nodes) -> Node:
    assert len(nodes) != 0, "list should not be empty"
    ancestor = low_com_ans(nodes[0], nodes[0])
    for i in range(1, len(nodes)):
        ancestor = low_com_ans(ancestor, nodes[i])
    return ancestor

def low_com_ans(node1: Node, node2: Node) -> Node:
    if (node1.level == node2.level):
        if (node1.name == node2.name):
            return node1
        else:
            return low_com_ans(node1.parent, node2.parent)
    elif (node1.level > node2.level):
        return low_com_ans(node1.parent, node2)
    else:
        return low_com_ans(node1, node2.parent)


def anon_bottomup(raw_dataset_file: str, DGH_folder: str, k: int,
                        output_file: str):
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    max_depths = DGH_maxdep(DGHs)
    sample_person = raw_dataset[0]
    person_att_list = list(sample_person.keys())
    att_list = list(DGHs.keys())
    quasi_atts = []
    sensitive_atts = []
    for att in person_att_list:
        if att in att_list:
            quasi_atts.append(att)
        else:
            sensitive_atts.append(att)

    levels = []
    for att in quasi_atts:
        levels.append(max_depths[att])

    levels_arg = list(map(lambda x: range(x + 1), levels))
    comb = list(itertools.product(*levels_arg))
    comb = sorted(comb, key=lambda x: sum(x))

    results = []
    result_lattice = float('inf')
    for t in comb:
        lattice = sum(t)
        if lattice > result_lattice:
            break
        anon_dataset = []
        for d in raw_dataset:
            anon_dataset.append(d.copy())
        for i in range(len(quasi_atts)):
            for person in anon_dataset:
                while DGHs[quasi_atts[i]][person[quasi_atts[i]]].level > (levels[i] - t[i]):
                    person[quasi_atts[i]] = DGHs[quasi_atts[i]][person[quasi_atts[i]]].parent.name
        k_anon = kanon_check(anon_dataset, k, sensitive_atts)
        if k_anon:
            cost = file_costLM(raw_dataset, anon_dataset, DGH_folder)
            results.append((cost, anon_dataset))
            result_lattice = lattice

    assert len(results) != 0, "k-anonymity cannot be satisfied"
    results = sorted(results)
    anonymized_dataset = results[0][1]

    write_dataset(anonymized_dataset, output_file)
def anon_random(raw_dataset_file: str, DGH_folder: str, k: int,
                      output_file: str, s: int):
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    for i in range(len(raw_dataset)):  ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s)  ## to ensure consistency between runs
    np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize

    clusters = []

    D = len(raw_dataset)

    count = D
    for i in range(D // k):
        if i == (D // k) - 1:
            clusters.append(raw_dataset[D - count:])
        else:
            clusters.append(raw_dataset[D - count:D - count + k])
        count -= k

    person_att_list = list(raw_dataset[0].keys())
    att_list = list(DGHs.keys())
    att_dict = {}
    att_anon_dict = {}

    for cluster in clusters:
        for att in person_att_list:
            if att in att_list:
                att_dict[att] = []
                att_anon_dict[att] = []
        for person in cluster:
            for att in list(att_dict.keys()):
                att_dict[att].append(person[att])
        for att in list(att_dict.keys()):
            att_val_list = att_dict[att]
            for i in range(len(att_val_list)):
                att_val_list[i] = DGHs[att][att_val_list[i]]
            att_anon_dict[att] = low_com_ans_lis(att_val_list)
        for person in cluster:
            for att in list(att_anon_dict.keys()):
                person[att] = att_anon_dict[att].name

    # END OF STUDENT'S CODE. Do not modify code in this function below this line.

    anonymized_dataset = [None] * D

    for cluster in clusters:  # restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)


def anon_clustering(raw_dataset_file: str, DGH_folder: str, k: int,
                          output_file: str):
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    firsts = DGH_firstele(DGH_folder)
    att_list = list(DGHs.keys())

    visited = [0] * len(raw_dataset)
    index = 0
    clusters = []
    while nonvisnum(visited) >= 2 * k:
        tuple_list = []
        cluster = []
        if visited[index] == 0:
            visited[index] = 1
            cluster.append(raw_dataset[index])
            person_att_list = list(raw_dataset[index].keys())
            for j in range(index + 1, len(raw_dataset)):
                if visited[j] == 0:
                    dist = 0
                    for i in person_att_list:
                        if i in att_list:
                            first = firsts[i]
                            dist += dist_cal(DGHs[i][raw_dataset[index][i]], DGHs[i][raw_dataset[j][i]], first)
                    tuple_list.append((dist, j))
            tuple_list = sorted(tuple_list)
            tuple_list = tuple_list[:(k - 1)]
            for t in tuple_list:
                visited[t[1]] = 1
                cluster.append(raw_dataset[t[1]])
            clusters.append(cluster)
        index += 1
    last_cluster = []
    for i in range(len(visited)):
        if visited[i] == 0:
            last_cluster.append(raw_dataset[i])
    if len(last_cluster) != 0:
        clusters.append(last_cluster)

    person_att_list = list(raw_dataset[0].keys())
    att_dict = {}
    att_anon_dict = {}

    for cluster in clusters:
        for att in person_att_list:
            if att in att_list:
                att_dict[att] = []
                att_anon_dict[att] = []
        for person in cluster:
            for att in list(att_dict.keys()):
                att_dict[att].append(person[att])
        att_dict_key_list = list(att_dict.keys())
        for att in range(len(att_dict_key_list)):
            att_val_list = att_dict[att_dict_key_list[att]]
            for i in range(len(att_val_list)):
                att_val_list[i] = DGHs[att_dict_key_list[att]][att_val_list[i]]
            att_anon_dict[att_dict_key_list[att]] = low_com_ans_lis(att_val_list)
        for person in cluster:
            for att in list(att_anon_dict.keys()):
                person[att] = att_anon_dict[att].name

    # Finally, write dataset to a file
    write_dataset(raw_dataset, output_file)



def DGH_firstele(DGH_folder: str):
    first_elements = {}
    DGHs = read_DGHs(DGH_folder)

    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        with open(DGH_file) as f:
            first_line = f.readline()
        first_line = "".join(first_line.split())
        first_elements[attribute_name] = DGHs[attribute_name][first_line]
    return first_elements


def DGH_maxdep(DGHs):
    max_depths = {}
    for att in list(DGHs.keys()):
        max_level = 0
        for val in list(DGHs[att].keys()):
            node = DGHs[att][val]
            if node.level > max_level:
                max_level = node.level
        max_depths[att] = max_level
    return max_depths


def nonvisnum(arr) -> int:
    counter = 0
    for i in range(len(arr)):
        if arr[i] == 0:
            counter += 1
    return counter


def dist_cal(node1: Node, node2: Node, first: Node):
    lca = low_com_ans(node1, node2)
    raw1 = (find_leaf_number(node1) - 1) / (find_leaf_number(first) - 1)
    anon1 = (find_leaf_number(lca) - 1) / (find_leaf_number(first) - 1)
    cost_LM1 = abs(raw1 - anon1)

    raw2 = (find_leaf_number(node2) - 1) / (find_leaf_number(first) - 1)
    anon2 = (find_leaf_number(lca) - 1) / (find_leaf_number(first) - 1)
    cost_LM2 = abs(raw2 - anon2)
    return cost_LM1 + cost_LM2

def read_dataset(dataset_file: str):
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    return result

def kanon_check(dataset, k, sensitive) -> bool:
    dataset_copy = []
    for person in dataset:
        person_copy = person.copy()
        dataset_copy.append(person_copy)

    for d in dataset_copy:
        for s in sensitive:
            d.pop(s)

    for d in dataset_copy:
        if dataset_copy.count(d) < k:
            return False
    return True


def write_dataset(dataset, dataset_file: str) -> bool:
    assert len(dataset) > 0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True

def read_DGHs(DGH_folder: str) -> dict:
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file)

    return DGHs

def read_DGH(DGH_file: str):
    dgh = {}
    f = open(DGH_file, 'r')

    prev_node = None

    for row in f.readlines():
        name = "".join(row.split())
        level = row.count('\t')
        node = Node(name, level)

        if level != 0:
            if level == prev_node.level:
                node.parent = prev_node.parent
            elif level == prev_node.level + 1:
                node.parent = prev_node
            else:
                prev_parent = prev_node.parent
                while prev_parent.level != level:
                    prev_parent = prev_parent.parent
                node.parent = prev_parent.parent
            node.parent.add_child(node)

        dgh[name] = node
        prev_node = node

    return dgh
