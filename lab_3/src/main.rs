use std::cell::RefCell;
use std::fmt::Debug;
use std::io::Read;
use std::ops::{Div, Mul };
use std::rc::Rc;
use derive_more::{Sub, Add, AddAssign, SubAssign};
use rand::distributions::Uniform;
use crate::delay_gen::DelayGen;

pub mod delay_gen {
    use rand::distributions::Distribution;
    use crate::TimeSpan;
    use rand::thread_rng;
    use rand_distr::{Exp, Normal, Uniform};

    #[derive(Clone, Copy, Debug)]
    pub enum DelayGen {
        Normal(Normal<f64>),
        Uniform(Uniform<f64>),
        Exponential(Exp<f64>),
    }

    impl DelayGen {
        pub fn sample(&self) -> TimeSpan {
            TimeSpan(
                match self {
                    Self::Normal(dist) => dist.sample(&mut thread_rng()).round() as u64,
                    Self::Uniform(dist) => dist.sample(&mut thread_rng()).round() as u64,
                    Self::Exponential(dist) => dist.sample(&mut thread_rng()).round() as u64,
                }
            )
        }
    }
}

#[derive(Debug, Copy, Clone, Default, Ord, PartialOrd, Eq, PartialEq)]
struct TimePoint(u64);

impl Sub for TimePoint {
    type Output = TimeSpan;
    fn sub(self, rhs: TimePoint) -> TimeSpan {
        TimeSpan(self.0 - rhs.0)
    }
}

impl Add<TimeSpan> for TimePoint {
    type Output = TimePoint;
    fn add(self, rhs: TimeSpan) -> TimePoint {
        TimePoint(self.0 + rhs.0)
    }
}

#[derive(Debug, Copy, Clone, Default, AddAssign)]
struct TimeSpan(u64);

#[derive(
    Debug, Copy, Clone, Default,
    Ord, PartialOrd, Eq, PartialEq,
    AddAssign, SubAssign, Sub
)]
struct QueueSize(u64);

impl Mul<TimeSpan> for QueueSize {
    type Output = QueueTimeDur;

    fn mul(self, rhs: TimeSpan) -> Self::Output {
        QueueTimeDur(self.0 * rhs.0)
    }
}

#[derive(Debug, Copy, Clone, Default, Add, AddAssign)]
struct QueueTimeDur(u64);

#[derive(Debug, Copy, Clone, Default)]
struct MeanQueueSize(f64);

impl Div<TimeSpan> for QueueTimeDur {
    type Output = MeanQueueSize;

    fn div(self, rhs: TimeSpan) -> Self::Output {
        MeanQueueSize(self.0 as f64 / rhs.0 as f64)
    }
}

impl Add<TimeSpan> for TimeSpan {
    type Output = TimeSpan;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

#[derive(Debug)]
struct Payload();

#[derive(Debug, Default)]
struct Cashier {
    queue_size: QueueSize,
    is_busy: bool,
}

#[derive(Debug)]
struct Bank {
    cashiers: [Cashier; 2],
    balance_count: BalancedCount,
    refused_count: RefusedCount,
    clients_count: ClientsCount
}

#[derive(Debug, Clone, Copy)]
enum CashierIndex {
    First,
    Second,
}

impl Bank {
    fn get_cashier_mut(&mut self, index: CashierIndex) -> &mut Cashier {
        match index {
            CashierIndex::First => &mut self.cashiers[0],
            CashierIndex::Second => &mut self.cashiers[1],
        }
    }

    fn get_cashier(&self, index: CashierIndex) -> &Cashier {
        match index {
            CashierIndex::First => &self.cashiers[0],
            CashierIndex::Second => &self.cashiers[1],
        }
    }
}

#[derive(Debug, Copy, Clone, AddAssign)]
struct RefusedCount(usize);

#[derive(Debug, Copy, Clone, AddAssign)]
struct BalancedCount(usize);

const QUEUE_CHANGE_SIZE: QueueSize = QueueSize(2);
const QUEUE_MAX_SIZE: QueueSize = QueueSize(3);

fn balance_queues(
    mut first_queue_size: QueueSize,
    mut second_queue_size: QueueSize,
) -> (QueueSize, QueueSize, BalancedCount) {
    let (mmin, mmax) = if first_queue_size < second_queue_size {
        (&mut first_queue_size, &mut second_queue_size)
    } else {
        (&mut second_queue_size, &mut first_queue_size)
    };
    let mut rebalanced_count = BalancedCount(0);
    while *mmax - *mmin >= QUEUE_CHANGE_SIZE {
        *mmin += QueueSize(1);
        *mmax -= QueueSize(1);
        rebalanced_count += BalancedCount(1);
    }
    (first_queue_size, second_queue_size, rebalanced_count)
}

#[derive(Debug, Clone, Copy, AddAssign)]
struct ClientsCount(usize);

#[derive(Debug)]
struct EventCreate {
    current_t: TimePoint,
    create_delay_gen: DelayGen,
    process_delay_gen: DelayGen,
    bank: Rc<RefCell<Bank>>,
}

#[derive(Debug)]
struct EventProcess {
    delay_gen: DelayGen,
    current_t: TimePoint,
    bank: Rc<RefCell<Bank>>,
    cashier_index: CashierIndex
}

impl EventCreate {
    fn iterate(self) -> (EventCreate, Option<EventProcess>) {
        self.bank.borrow_mut().clients_count += ClientsCount(1);
        let event_finish_process_time = self.current_t + self.process_delay_gen.sample();
        let bank_clone = self.bank.clone();
        let is_first_queue_busy = self.bank.borrow_mut()
            .get_cashier(CashierIndex::First).is_busy;
        let is_second_queue_busy = self.bank.borrow_mut()
            .get_cashier(CashierIndex::Second).is_busy;
        (
            EventCreate {
                current_t: self.current_t + self.create_delay_gen.sample(),
                create_delay_gen: self.create_delay_gen,
                process_delay_gen: self.process_delay_gen,
                bank: self.bank.clone()
            },
            match (
                is_first_queue_busy,
                is_second_queue_busy
            ) {
                (false, false) => {
                    let mut bank = self.bank.borrow_mut();
                    assert_eq!(bank.get_cashier(CashierIndex::First).queue_size, QueueSize(0));
                    assert_eq!(bank.get_cashier(CashierIndex::Second).queue_size, QueueSize(0));
                    bank.get_cashier_mut(CashierIndex::First).is_busy = true;
                    Some(EventProcess {
                        delay_gen: self.process_delay_gen,
                        current_t: event_finish_process_time,
                        bank: bank_clone,
                        cashier_index: CashierIndex::First
                    })
                },
                (true, false) => {
                    let mut bank = self.bank.borrow_mut();
                    assert_eq!(bank.get_cashier(CashierIndex::Second).queue_size, QueueSize(0));
                    bank.get_cashier_mut(CashierIndex::Second).is_busy = true;
                    Some(EventProcess {
                        delay_gen: self.process_delay_gen,
                        current_t: event_finish_process_time,
                        bank: bank_clone,
                        cashier_index: CashierIndex::Second
                    })
                },
                (false, true) => {
                    let mut bank = self.bank.borrow_mut();
                    assert_eq!(bank.get_cashier(CashierIndex::First).queue_size, QueueSize(0));
                    bank.get_cashier_mut(CashierIndex::First).is_busy = true;
                    Some(EventProcess {
                        delay_gen: self.process_delay_gen,
                        current_t: event_finish_process_time,
                        bank: bank_clone,
                        cashier_index: CashierIndex::First
                    })
                }
                (true, true) => {
                    let cashier_index = {
                        let mut bank = self.bank.borrow_mut();
                        let first_queue_size = bank.get_cashier(CashierIndex::First).queue_size;
                        let second_queue_size = bank.get_cashier(CashierIndex::Second).queue_size;
                        if first_queue_size < second_queue_size {
                            CashierIndex::First
                        } else {
                            CashierIndex::Second
                        }
                    };
                    if self.bank.borrow().get_cashier(cashier_index).queue_size >= QUEUE_MAX_SIZE {
                        self.bank.borrow_mut().refused_count += RefusedCount(1);
                        None
                    } else {
                        let mut bank = self.bank.borrow_mut();
                        bank.get_cashier_mut(cashier_index).queue_size += QueueSize(1);
                        let res = balance_queues(
                            bank.get_cashier(CashierIndex::First).queue_size,
                            bank.get_cashier(CashierIndex::Second).queue_size,
                        );
                        bank.get_cashier_mut(CashierIndex::First).queue_size = res.0;
                        bank.get_cashier_mut(CashierIndex::Second).queue_size = res.1;
                        bank.balance_count += res.2;

                        Some(EventProcess {
                            delay_gen: self.process_delay_gen,
                            current_t: event_finish_process_time,
                            bank: bank_clone,
                            cashier_index
                        })
                    }
                }
            }
        )
    }
}

impl EventProcess {
    fn iterate(self) -> Option<EventProcess> {
        let mut bank = self.bank.borrow_mut();
        let queue_size = bank.get_cashier(self.cashier_index).queue_size;
        bank.get_cashier_mut(self.cashier_index).is_busy = false;
        if queue_size > QueueSize(0) {
            bank.get_cashier_mut(self.cashier_index).queue_size -= QueueSize(1);

            let res = balance_queues(
                bank.get_cashier(CashierIndex::First).queue_size,
                bank.get_cashier(CashierIndex::Second).queue_size,
            );
            bank.get_cashier_mut(CashierIndex::First).queue_size = res.0;
            bank.get_cashier_mut(CashierIndex::Second).queue_size = res.1;
            bank.balance_count += res.2;

            bank.get_cashier_mut(self.cashier_index).is_busy = true;
            Some(EventProcess {
                delay_gen: self.delay_gen,
                current_t: self.current_t + self.delay_gen.sample(),
                bank: self.bank.clone(),
                cashier_index: self.cashier_index
            })
        } else {
            None
        }
    }
}

#[derive(Debug)]
enum Event {
    EventProcess(EventProcess),
    EventCreate(EventCreate),
}

impl Event {
    fn get_current_t(&self) -> TimePoint {
        match self {
            Event::EventProcess(e) => e.current_t,
            Event::EventCreate(e) => e.current_t,
        }
    }
}

fn main() {
    let bank = Rc::new(RefCell::new(
        Bank {
            cashiers: Default::default(),
            balance_count: BalancedCount(0),
            refused_count: RefusedCount(0),
            clients_count: ClientsCount(0),
        }
    ));
    let mut nodes = vec![
        Event::EventCreate(EventCreate {
            current_t: Default::default(),
            create_delay_gen: DelayGen::Uniform(Uniform::new(0.0, 10.0)),
            process_delay_gen: DelayGen::Uniform(Uniform::new(40.0, 50.0)),
            bank,
        })
    ];
    let total_time = loop {
        nodes.sort_by(|a, b| b.get_current_t().cmp(&a.get_current_t()));
        let next_event = nodes.pop().unwrap();
        if next_event.get_current_t() > TimePoint(1000) {
            break next_event.get_current_t();
        }
        // dbg!(&next_event);
        match next_event {
            Event::EventCreate(event) => {
                let (event_create, event_process) = event.iterate();
                nodes.push(Event::EventCreate(event_create));
                if let Some(event_process) = event_process {
                    nodes.push(Event::EventProcess(event_process));
                }
            },
            Event::EventProcess(event) => {
                if let Some(event_process) = event.iterate() {
                    nodes.push(Event::EventProcess(event_process));
                }
            }
        }
    };
    let node = nodes.pop().unwrap();
    let bank = match node {
        Event::EventCreate(event) => event.bank,
        Event::EventProcess(event) => event.bank
    };
    let bank = bank.borrow();

    println!("refused_count: {:?}", bank.refused_count.0 as f64 / bank.clients_count.0 as f64);
    println!("balance_count: {:?}", bank.balance_count);
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_bank() {

    }
}


