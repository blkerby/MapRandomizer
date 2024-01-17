use hashbrown::HashMap;

pub struct SuffixTree {
    pub tree: Tree,
    pub data: Vec<u8>,
}

impl SuffixTree {
    pub fn new(data: Vec<u8>) -> Self {
        SuffixTree {
            tree: Tree::new(&data),
            data,
        }
    }

    pub fn lookup(&self, query: &[u8]) -> (usize, usize) {
        self.tree.lookup(query, &self.data)
    }
}

#[derive(PartialEq, Eq, Debug)]
enum Tree {
    Leaf(usize),  // address 
    Node(HashMap<Option<u8>, Tree>),  // children by first byte
}

impl Tree {
    pub fn new(data: &[u8]) -> Self {
        let mut tree = Tree::Leaf(0);
        for i in 1..data.len() {
            println!("{}", i);
            Self::add_suffix(&mut tree, data, i);
        }
        tree
    }

    fn add_suffix(mut tree: &mut Tree, data: &[u8], mut idx: usize) {
        loop {
            assert!(idx <= data.len());
            let b: Option<u8> = if idx == data.len() {
                None
            } else {
                Some(data[idx])
            };
            match tree {
                &mut Tree::Leaf(addr) => {
                    if Some(data[addr]) == b {
                        let mut new_map = HashMap::new();
                        new_map.insert(b, Tree::Leaf(addr + 1));
                        *tree = Tree::Node(new_map);
                        tree = match tree {
                            Tree::Node(m) => {
                                m.get_mut(&b).unwrap()
                            },
                            _ => panic!("Unexpected error in add_prefix")
                        };
                        idx += 1;
                        continue;
                    } else {
                        let mut new_map: HashMap<Option<u8>, Tree> = HashMap::new();
                        new_map.insert(Some(data[addr]), Tree::Leaf(addr + 1));
                        new_map.insert(b, Tree::Leaf(idx + 1));
                        *tree = Tree::Node(new_map);
                        break;
                    }
                },
                Tree::Node(children) => {
                    if children.contains_key(&b) {
                        tree = children.get_mut(&b).unwrap();
                        idx += 1;
                        continue;
                    } else {
                        children.insert(b, Tree::Leaf(idx + 1));
                        break;                            
                    }
                }
            }
        }
    }

    // Returns address and length of longest match.
    fn lookup(&self, mut query: &[u8], data: &[u8]) -> (usize, usize) {
        let mut tree = self;
        let mut length = 0;
        loop {
            match tree {
                &Tree::Leaf(addr) => {
                    let max_len = std::cmp::min(query.len(), data.len() - addr);
                    for i in 0..max_len {
                        if query[i] != data[addr + i] {
                            return (addr - length, length + i);
                        }
                    }
                    return (addr - length, length + max_len);
                },
                Tree::Node(children) => {
                    if query.len() > 0 && children.contains_key(&Some(query[0])) {
                        tree = children.get(&Some(query[0])).unwrap();
                        length += 1;
                        query = &query[1..];
                    } else {
                        // We've found the longest possible match(es). Now just pick one.
                        let mut skip_length = 1;
                        tree = children.iter().next().unwrap().1;
                        loop {
                            match tree {
                                &Tree::Leaf(addr) => {
                                    return (addr - length - skip_length, length);
                                },
                                Tree::Node(children) => {
                                    tree = children.iter().next().unwrap().1;
                                    skip_length += 1;
                                },
                            }
                        }

                    }
                }
            }    
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1() {
        let data = [10, 20, 10];
        let tree = Tree::new(&data);
        assert_eq!(tree.lookup(&[20], &data), (1, 1));
        assert_eq!(tree.lookup(&[10, 20], &data), (0, 2));
        assert_eq!(tree.lookup(&[20, 10, 30], &data), (1, 2));
        assert_eq!(tree.lookup(&[10, 20, 10], &data), (0, 3));
    }

    #[test]
    fn test_2() {
        let data = [10, 20, 30, 20, 50, 20, 50, 70];
        let tree = Tree::new(&data);
        assert_eq!(tree.lookup(&[30, 20, 50, 60], &data), (2, 3));
        assert_eq!(tree.lookup(&[20, 50, 20, 80], &data), (3, 3));
    }
}
