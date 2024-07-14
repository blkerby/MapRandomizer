use hashbrown::HashMap;

type DataIdx = u32;
type NodeId = u32;
type EdgeId = u8;

const DATA_END: DataIdx = DataIdx::MAX;
const NODE_UNDEFINED: NodeId = NodeId::MAX;
const EDGE_UNDEFINED: EdgeId = EdgeId::MAX; // Note: This is not a distinct value from valid edge IDs.

// We build SuffixTree using Ukkonen's algorithm, appending one byte at a time, starting from an empty data vector.
// This is a compact way indexing substrings of a data vector (Vec<u8>) in a way that allows efficient searching.
// Even though there are O(n^2) substrings, the tree structure only occupies O(n) space and takes O(n) time to construct.
// Every node in the tree represents a substring that occurs in the data (possibly in multiple places); the substring
// represented by a node is given by concatenating the edge label of all the edges along the path from the root node
// down to the given node. Only internal nodes (not leaf nodes) are explicitly represented using a Node struct.
// At each stage, the following invariants are satisfied:
// 1. Every possible substring of the data (i.e. data[i..j] for any i <= j) is represented somewhere in the tree, meaning
//    there is a path starting from the root node to another node, such that concatenating the edge labels along the path
//    (with possibly just a prefix of the last edge label being used) results in that substring.
// 2. For every node and every possible byte value, the node has at most one child edge with label starting with that byte value.
// 3. Every (non-leaf) node has at least two child edges, except for possibly the root node.
// 4. A special "cut" position is kept track of, such that for all i < cut, the suffix data[i..] does not occur as a
//    substring anywhere else in the data (which means it is represented by a leaf).
// 5. For every (non-leaf) non-root node representing data[i..j] (for i < j), a "suffix link" is maintained, which is a pointer to
//    the node for data[(i + 1)..j].
#[derive(Debug)]
pub struct SuffixTree {
    pub data: Vec<u8>,
    nodes: Vec<Node>,
    // position of smallest suffix data[i..] such that for all j < i, the substring data[j..] occurs nowhere else in data:
    // Note we don't track the index "i" explicitly (as it is not needed), only the position in the tree.
    cut: Position,
}

// Identifies a position in the tree by a node and (optionally) a prefix of a leaf child edge belonging to it
#[derive(Debug)]
struct Position {
    node_id: NodeId,
    edge_id: EdgeId, // index of child edge below this node (ignored if length == 0).
    length: DataIdx, // length of prefix of the child edge label to follow
}

#[derive(Debug)]
struct Node {
    edges: Vec<Edge>,             // Child edges below this node
    edge_lookup: HashMap<u8, u8>, // Mapping from first byte of edge label to index in `edges`.
    suffix_link: NodeId,
}

#[derive(Debug)]
struct Edge {
    // Edge label references data indirectly, so that it takes constant space regardless of its length.
    // If the edge ends in a leaf (if `node_id == NODE_UNDEFINED`), then it references data[start..].
    // Otherwise it references data[start..end].
    start: DataIdx,  // Starting index (inclusive) of referenced data
    end: DataIdx,    // Ending index (exclusive) of referenced data
    node_id: NodeId, // ID of child node that the edge points to. This is a special value NODE_UNDEFINED for a leaf node.
}

impl SuffixTree {
    pub fn new(data: &[u8]) -> Self {
        let root_node = Node {
            edges: Vec::new(),
            edge_lookup: HashMap::new(),
            suffix_link: NODE_UNDEFINED,
        };
        let mut tree = SuffixTree {
            data: vec![],
            nodes: vec![root_node],
            cut: Position {
                node_id: 0,
                edge_id: EDGE_UNDEFINED,
                length: 0,
            },
        };
        for &b in data {
            tree.push_byte(b);
        }
        tree
    }

    // Returns the index (into `self.data`) and length of a longest-matching prefix of `query`.
    pub fn find_longest_prefix(&self, query: &[u8]) -> (DataIdx, DataIdx) {
        let mut node_id = 0; // Start at root node
        let mut pos: DataIdx = 0; // Length of prefix of query that has been matched so far
        if query.len() == 0 {
            // Handle trivial edge case:
            return (0, 0);
        }
        loop {
            let node = &self.nodes[node_id as usize];
            let b = query[pos as usize];
            if let Some(&edge_id) = node.edge_lookup.get(&b) {
                let edge = &node.edges[edge_id as usize];
                for i in 0..edge.end - edge.start {
                    let whole_query_match = pos + i == query.len() as DataIdx;
                    let reached_data_end = edge.start + i == self.data.len() as DataIdx;
                    if whole_query_match
                        || reached_data_end
                        || self.data[(edge.start + i) as usize] != query[(pos + i) as usize]
                    {
                        // We are done: either the whole query matched, or we reached the data end or a point where it doesn't match.
                        return (edge.start - pos, pos + i);
                    }
                }

                // Entire edge matched, so continue.
                pos += edge.end - edge.start;
                node_id = edge.node_id;
                if pos == query.len() as DataIdx {
                    return (edge.end - query.len() as DataIdx, query.len() as DataIdx);
                }
            } else {
                // No further bytes match, so we're done.
                if node.edges.len() == 0 {
                    // Node has no child edges. The only way this should happen is at the root node, with self.data is empty.
                    assert!(node_id == 0 && self.data.len() == 0);
                    return (0, 0);
                }
                let edge = &node.edges[0]; // Pick an arbitrary edge
                return (edge.start - pos, pos);
            }
        }
    }

    // Follow data[start..(start + length)] starting from given node, and return the resulting position.
    fn follow_data(
        &self,
        mut node_id: NodeId,
        mut start: DataIdx,
        mut length: DataIdx,
    ) -> Position {
        loop {
            if length == 0 {
                return Position {
                    node_id,
                    edge_id: EDGE_UNDEFINED,
                    length: 0,
                };
            }
            let node = &self.nodes[node_id as usize];
            let b = self.data[start as usize];
            let edge_id = node.edge_lookup[&b];
            let edge = &node.edges[edge_id as usize];
            let edge_len = edge.end - edge.start;
            if length < edge_len || edge.end == DATA_END {
                return Position {
                    node_id,
                    edge_id,
                    length,
                };
            }
            start += edge_len;
            length -= edge_len;
            node_id = edge.node_id;
        }
    }

    pub fn push_byte(&mut self, b: u8) {
        // Append the byte to the data vector:
        self.data.push(b);

        // Now for every index i, we need to ensure that the substring data[i..] exists in the tree
        // (where data[i..] now includes the newly added byte at the end).
        // For i less than the cut position, this is already taken care of automatically, since in such cases
        // data[i..] is represented by a leaf, which has an implicit end position (DATA_END).
        loop {
            let cut = &self.cut;
            let num_nodes = self.nodes.len();
            let node = &mut self.nodes[cut.node_id as usize];
            let node_suffix_link = node.suffix_link;
            if cut.length == 0 {
                if node.edge_lookup.contains_key(&b) {
                    // data[cut..] already existed in the tree. This means data[i..] also already existed for i > cut.
                    // So we only have to update the cut position, and we're done.
                    self.cut.edge_id = node.edge_lookup[&b];
                    self.cut.length = 1;
                    let edge = &node.edges[self.cut.edge_id as usize];
                    if edge.end - edge.start == 1 {
                        self.cut.length = 0;
                        self.cut.node_id = edge.node_id;
                        self.cut.edge_id = EDGE_UNDEFINED;
                    }
                    return;
                } else {
                    // Add a new leaf edge containing the single new byte:
                    let edge_id = node.edges.len() as EdgeId;
                    let edge = Edge {
                        start: (self.data.len() - 1) as DataIdx,
                        end: DATA_END,
                        node_id: NODE_UNDEFINED,
                    };
                    node.edges.push(edge);
                    node.edge_lookup.insert(b, edge_id);

                    // With the new byte, data[cut..] now becomes a leaf, so advance the cut position.
                    if node.suffix_link == NODE_UNDEFINED {
                        // The node doesn't have a suffix link; this should only happen at the root node.
                        assert!(cut.node_id == 0); // Check that this is in fact the root node.
                                                   // Since cut.length == 0, the cut position is at the end (representing the empty string), so we are done.
                        self.cut.edge_id = edge_id;
                        self.cut.length = 1;
                        return;
                    } else {
                        // Follow the suffix link.
                        self.cut = Position {
                            node_id: node.suffix_link,
                            edge_id: EDGE_UNDEFINED,
                            length: 0,
                        };
                    }
                }
            } else {
                let edge = &mut node.edges[cut.edge_id as usize];
                let edge_start = edge.start;
                if edge.start + cut.length == (self.data.len() - 1) as DataIdx {
                    // We're at the end of a leaf node, which will be implicitly extended by the new byte.
                    assert!(edge.end == DATA_END);

                    // With the new byte, data[cut..] now becomes a leaf, so advance the cut position
                    // by following the suffix link, and populate the suffix link of the newly created node.
                    self.cut = if node_suffix_link == NODE_UNDEFINED {
                        // The node doesn't have a suffix link; this should only happen at the root node.
                        assert!(cut.node_id == 0); // Check that this is in fact the root node.
                                                   // Locate the new cut position directly, by stripping off the first byte of data and
                                                   // following a path down the tree, starting from the root.
                        let edge = &node.edges[cut.edge_id as usize];
                        let new_start = edge.start + 1;
                        let new_length = cut.length - 1;
                        self.follow_data(0, new_start, new_length)
                    } else {
                        self.follow_data(node_suffix_link, edge_start, cut.length)
                    };
                } else {
                    let c = self.data[(edge.start + cut.length) as usize];
                    if c == b {
                        // data[cut..] already existed in the tree. This means data[i..] also already existed for i > cut.
                        // So we only have to update the cut position, and we're done.
                        self.cut.length += 1;
                        if self.cut.length == edge.end - edge.start {
                            self.cut.length = 0;
                            self.cut.node_id = edge.node_id;
                            self.cut.edge_id = EDGE_UNDEFINED;
                        }
                        return;
                    } else {
                        // The edge must be split and a new node created in the middle:
                        let tail_edge = Edge {
                            start: edge.start + cut.length,
                            end: edge.end,
                            node_id: edge.node_id,
                        };
                        let leaf_edge = Edge {
                            start: (self.data.len() - 1) as DataIdx,
                            end: DATA_END,
                            node_id: NODE_UNDEFINED,
                        };
                        let mut new_node = Node {
                            edges: vec![tail_edge, leaf_edge],
                            edge_lookup: vec![(c, 0), (b, 1)].into_iter().collect(),
                            suffix_link: NODE_UNDEFINED,
                        };
                        let new_node_id = num_nodes as NodeId;
                        edge.end = edge.start + cut.length;
                        edge.node_id = new_node_id;

                        // With the new byte, data[cut..] now becomes a leaf, so advance the cut position
                        // by following the suffix link, and populate the suffix link of the newly created node.
                        let new_cut = if node_suffix_link == NODE_UNDEFINED {
                            // The node doesn't have a suffix link; this should only happen at the root node.
                            assert!(cut.node_id == 0); // Check that this is in fact the root node.
                                                       // Locate the new cut position directly, by stripping off the first byte of data and
                                                       // following a path down the tree, starting from the root.
                            let edge = &node.edges[cut.edge_id as usize];
                            let new_start = edge.start + 1;
                            let new_length = cut.length - 1;
                            self.follow_data(0, new_start, new_length)
                        } else {
                            self.follow_data(node_suffix_link, edge_start, cut.length)
                        };

                        if new_cut.length == 0 {
                            // A node for the new node's suffix link already exists.
                            new_node.suffix_link = new_cut.node_id;
                        } else {
                            // A node for the new node's suffix link doesn't yet exist but will be created on the next loop iteration.
                            // We already know what its ID will be.
                            new_node.suffix_link = (num_nodes + 1) as NodeId;
                        }
                        self.nodes.push(new_node);
                        self.cut = new_cut;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};

    use super::*;

    #[test]
    fn test_simple() {
        let data = [10, 20, 10, 20, 30, 10, 20];
        let tree = SuffixTree::new(&data);
        assert_eq!(tree.find_longest_prefix(&[10, 20, 30]), (2, 3));
        assert_eq!(tree.find_longest_prefix(&[20, 30]), (3, 2));
        assert_eq!(tree.find_longest_prefix(&[30, 50]), (4, 1));
        assert_eq!(tree.find_longest_prefix(&[10, 20, 10]), (0, 3));
        assert_eq!(tree.find_longest_prefix(&[20, 10, 40]), (1, 2));
    }

    // Check if query appears as substring in data:
    fn check_query(data: &[u8], query: &[u8]) -> bool {
        for w in data.windows(query.len()) {
            if w == query {
                return true;
            }
        }
        return false;
    }

    #[test]
    fn test_random() {
        let mut rng = rand::rngs::StdRng::from_seed([0u8; 32]);
        for _ in 0..100 {
            let mut data = vec![];
            let length = 1000;
            for _ in 0..length {
                data.push(rng.gen_range(0..4) as u8);
            }
            data.push(5);
            let tree = SuffixTree::new(&data);
            for query_length in 1..10 {
                let num_queries = 100;
                for _ in 0..num_queries {
                    let mut query = vec![];
                    for _ in 0..query_length {
                        query.push(rng.gen_range(0..4) as u8);
                    }
                    let (start, match_length) = tree.find_longest_prefix(&query);
                    let has_match = check_query(&data, &query);
                    assert_eq!(match_length == query_length, has_match);
                    let data_slice =
                        &data[start as usize..(start as usize + match_length as usize)];
                    let query_slice = &query[..match_length as usize];
                    assert_eq!(data_slice, query_slice);
                }
            }
        }
    }
}
