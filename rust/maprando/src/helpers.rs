use maprando_game::IndexedVec;

use crate::{randomize::ItemPriorityGroup, settings::{KeyItemPriority, KeyItemPrioritySetting}};

pub fn get_item_priorities(
    item_priorities: &[KeyItemPrioritySetting],
) -> Vec<ItemPriorityGroup> {
    let mut priorities: IndexedVec<KeyItemPriority> = IndexedVec::default();
    priorities.add(&KeyItemPriority::Early);
    priorities.add(&KeyItemPriority::Default);
    priorities.add(&KeyItemPriority::Late);

    let mut out: Vec<ItemPriorityGroup> = Vec::new();
    for &priority in &priorities.keys {
        out.push(ItemPriorityGroup {
            priority,
            items: vec![],
        });
    }
    for x in item_priorities {
        let i = priorities.index_by_key[&x.priority];
        out[i].items.push(format!("{:?}", x.item));
    }
    out
}
