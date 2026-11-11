// use utote::Multiset;
//
// trait Dual { fn dual() {} }
//
// trait Zero {}
// trait Nat {}
// // trait Bag { fn add(&self,other: &Self)->dyn Bag; }
// trait Poly {}
// trait Multi {}
//
// type N = usize;
//
// struct Empty {} impl Zero for Empty {}
// struct Full {} impl Zero for Full {}
// struct Number<const N:size> {}
//
// // type Output=Self;
// trait Multiset<T> { fn add(self,other:Self)->Self; }
// struct Bag<T> { items:Vec<T> }
// impl Multiset<T> for Bag<T> { fn add(self, other: Self) -> Self {
//     self.items.cop
//   } }
//
//
//
// mod tests {
//   use crate::bag::bag;
//
//   fn test1() {
//     type N = usize;
//
//
//
//
//     // A multiset of 5 elements, which can be counted up to u8::MAX
//     let empty = bag([false]);
//     let zero = Multiset::<empty,0>::from([]);
//     let one = Multiset::<empty,1>::from([empty]);
//     let two = Multiset::<empty,2>::from([empty,empty]);
//     let three = Multiset::<empty,3>::from([empty,empty,empty]);
//     let full = bag([true]);
//     let b0 = bag::<bool,0>([]);
//     let b5 = Bag::<u8, 5>.from();
//     let mut multiset = bag([0u8, 3, 4, 0, 5]);
//     let equivalent_multiset = Bag<u8, 5>([0, 3, 4, 0, 5]);
//     multiset.insert(2, 6);
//     assert_eq!(multiset.contains(0), false);
//     assert_eq!(multiset.contains(1), true);
//   }
//
// }
