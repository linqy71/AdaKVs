import dataclasses
from typing import List,Dict
from itertools import count
from collections import defaultdict

"""
maintain token's longest common prefix

RadixTree接口:
def insert(self,key:List[int]) 插入token序列,后续还可能要插入kv_ptr和重要性(注意一个性质,NR整体在最后一个新生成节点)
def search(self,key:List[int])->List[RadixToken] 查找最长公共前缀,返回数据类型为RadixToken。
"""
@dataclasses.dataclass
class CachePointer:
    # 通过chunk_id定位chunk，通过offset定位在chunk内的位置
    # chunk中的kv缓存为一个TorchTensor,存了chunk_size个token的KV缓存, shape为(chunk_size, b * n_head, head_dim)
    # offset表明该token的KV张量为 (offset:offset+1, 0:b * n_head, 0:head_dim)
    chunk_id: int
    offset: int

class RadixToken:
    token_count = count()
    def __init__(self, token_id:int, token_name = None, importance:int = 0, kv_ptr:List[CachePointer] = None):
        self.token_id = token_id

        self.kv_ptr = kv_ptr  # 为该token所有层的chunk_id. kv_ptr[i]表示第i层的缓存的CachePointer

        self.importance = importance

        self.token_name = token_name or RadixToken.next_token_name()

    def __repr__(self):
        return f"(id={self.token_id},imp={self.importance})"

    @classmethod
    def next_token_name(cls):
        return next(cls.token_count)

class RadixTreeNode:
    def __init__(self, prefix_id, tokens:List[RadixToken], mapping_list=None):
        # tokens始终保留原来的顺序,重排只需要修改mapping_list,要获取重排后的顺序使用mapping_list
        # 即 tokens[mapping_list[i]] 表示按照重要性排序后的, 第i个token
        self.prefix_id = prefix_id
        self.tokens = tokens
        if mapping_list is not None:
            self.mapping_list = mapping_list
        else:
            self.mapping_list = [i for i in range(len(tokens))]

        self.children:Dict[int,RadixTreeNode] = {}

    def __repr__(self):
        return f"Node(prefix_id={self.prefix_id}, token={self.tokens}, mapping_list={self.mapping_list})"

    def sort_by_importance(self):
        self.mapping_list.sort(key=lambda i: self.tokens[i].importance, reverse=True)

    def get_sorted_tokens(self):
        return [self.tokens[i] for i in self.mapping_list]

    @classmethod
    def create_from_token_id(cls, prefix_id, tokens:List[int], kv_ptr=None):
        if kv_ptr is None:
            tokens = [RadixToken(token_id=x) for x in tokens]
        else:
            tokens = [RadixToken(token_id=tokens[i], kv_ptr=kv_ptr[i]) for i in range(len(tokens))]
        return cls(prefix_id, tokens)



class RadixTree:
    def __init__(self):
        # root is an empty node
        self.root = RadixTreeNode(prefix_id=0, tokens=[])
        self.max_prefix_id = 0

    # 计算两个序列的最长公共前缀长度
    @staticmethod
    def common_prefix_length(a:List, b:List[RadixToken]):
        min_len = min(len(a),len(b))
        for i in range (min_len):
            if a[i] != b[i].token_id:
                return i;
        return min_len

    # 插入新的序列,同时插入新节点的kv_ptr
    # kv_ptr:List[List[CachePointer]]  kv_ptr[i]表示NR中第i个token在每一层的缓存的位置
    # 返回new_prefix_id, 表示需要将新的prefix映射到lsh
    def insert(self, key:List[int], kv_ptr = None):
        if len(key) == 0:
            return
        current = self.root
        remaining = key

        new_prefix_id = 0

        while remaining:
            next_token = remaining[0]
            if next_token in current.children:
                child = current.children[next_token]
                common_len = self.common_prefix_length(remaining,child.tokens)

                if common_len == len(child.tokens):
                    remaining = remaining[common_len:]
                    current = child
                else:
                    # split node, remaining the same prefix_id
                    new_node = RadixTreeNode(prefix_id=child.prefix_id, tokens=child.tokens[:common_len])
                    new_node.mapping_list = [x for x in child.mapping_list if x < common_len]

                    child.tokens = child.tokens[common_len:]
                    child.mapping_list = [x-common_len for x in child.mapping_list if x >= common_len]

                    current.children[next_token] = new_node
                    new_node.children[child.tokens[0].token_id] = child

                    remaining = remaining[common_len:]
                    current = new_node
            else:
                self.max_prefix_id += 1
                new_prefix_id = self.max_prefix_id
                new_node = RadixTreeNode.create_from_token_id(self.max_prefix_id, remaining, kv_ptr)
                current.children[next_token] = new_node
                break

        return new_prefix_id


    # 查询最长公共前缀，返回最长公共前缀的kv_ptr
    def search(self, key:List[int]):
        current = self.root
        remaining = key
        ans = list()
        while remaining:
            next_token = remaining[0]
            if next_token in current.children:
                child = current.children[next_token]
                common_len = self.common_prefix_length(remaining,child.tokens)
                ans.extend(
                    [t.kv_ptr for t in child.tokens[:common_len]]
                )
                if common_len != len(child.tokens):
                    break
                remaining = remaining[common_len:]
                current = child
            else:
                break
        # type = List[List[CachePointer]]
        return ans

    # 查询最长公共前缀，返回最长公共前缀的prefix_id和对应prefix长度
    # 每个prefix_id代表对应的tokens被映射到一个LSH，每个prefix_id可能包含多个node
    # 每个prefix_id对应的max_common_len用于确保在LSH中查询时不会查询到超过max_common_len的token
    def match(self, key:List[int]):
        current = self.root
        remaining = key
        matched_prefix = defaultdict(int)
        while remaining:
            next_token = remaining[0]
            if next_token in current.children:
                child = current.children[next_token]
                common_len = self.common_prefix_length(remaining,child.tokens)
                matched_prefix[child.prefix_id] += common_len
                if common_len != len(child.tokens):
                    break
                remaining = remaining[common_len:]
                current = child
            else:
                break
        # type = List[List[CachePointer]]
        return matched_prefix

    def delete(self, key:List):
        pass

    def visualize(self, node=None, depth=0):
        """for debug"""
        if node is None:
            node = self.root
            print("root:")
        else:
            indent = "  " * depth
            print(f"{indent}{node}")

        for child in node.children.values():
            self.visualize(child, depth + 1)
