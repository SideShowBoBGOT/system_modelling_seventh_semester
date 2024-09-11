mod task_2;
mod utils;

use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::{Div, Mul };
use std::rc::Rc;
use derive_more::{Sub, Add, AddAssign, SubAssign};
use rand_distr::Exp;
use scopeguard::defer;
use crate::utils::delay_gen::DelayGen;
use crate::utils::TimePoint;

extern crate scopeguard;



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
pub struct TimeSpan(f64);

#[derive(
    Debug, Copy, Clone, Default,
    Ord, PartialOrd, Eq, PartialEq,
    AddAssign, SubAssign, Sub
)]
struct QueueSize(u64);

impl Mul<TimeSpan> for QueueSize {
    type Output = QueueTimeDur;

    fn mul(self, rhs: TimeSpan) -> Self::Output {
        QueueTimeDur(self.0 as f64 * rhs.0)
    }
}

#[derive(Debug, Copy, Clone, Default, Add, AddAssign)]
struct QueueTimeDur(f64);

#[derive(Debug)]
struct FloatQueueSize(f64);

impl Div<TimeSpan> for QueueTimeDur {
    type Output = FloatQueueSize;

    fn div(self, rhs: TimeSpan) -> Self::Output {
        FloatQueueSize(self.0 / rhs.0)
    }
}

impl Add<TimeSpan> for TimeSpan {
    type Output = TimeSpan;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

#[derive(Debug, Default)]
struct Cashier {
    queue_size: QueueSize,
    is_busy: CashierBusy,

    processed_clients: ClientsCount,
    work_time: TimeSpan,

    queue_time_span: QueueTimeDur,
}

#[derive(Debug, Clone, Copy)]
enum CashierIndex {
    First,
    Second,
}

#[derive(Debug, Clone, Copy, Default, AddAssign)]
struct BusyCashiersCount(f64);

impl Mul<TimeSpan> for BusyCashiersCount {
    type Output = BusyCashiersCountTimeSpan;
    fn mul(self, rhs: TimeSpan) -> Self::Output {
        BusyCashiersCountTimeSpan(self.0 * rhs.0 as f64)
    }
}

#[derive(Debug, Clone, Copy, Default, AddAssign)]
struct BusyCashiersCountTimeSpan(f64);

impl Div<TimeSpan> for BusyCashiersCountTimeSpan {
    type Output = BusyCashiersCount;
    fn div(self, rhs: TimeSpan) -> Self::Output {
        BusyCashiersCount(self.0 / rhs.0 as f64)
    }
}

#[derive(Debug, Clone, Copy, Default, AddAssign)]
struct ClientsCountTimeSpan(f64);

impl Mul<TimeSpan> for ClientsCount {
    type Output = ClientsCountTimeSpan;

    fn mul(self, rhs: TimeSpan) -> Self::Output {
        ClientsCountTimeSpan(self.0 as f64 * rhs.0)
    }
}

struct FloatClientsCount(f64);

impl Div<TimeSpan> for ClientsCountTimeSpan {
    type Output = FloatClientsCount;
    fn div(self, rhs: TimeSpan) -> Self::Output {
        FloatClientsCount(self.0 / rhs.0)
    }
}

#[derive(Debug, Default)]
struct Bank {
    cashiers: [Cashier; 2],
    balance_count: BalancedCount,
    refused_count: RefusedCount,
    clients_count: ClientsCount,

    create_event_time_span: ClientsCountTimeSpan,
    last_create_event_time: TimePoint,

    busy_cashiers_count_time_span: BusyCashiersCountTimeSpan,
    event_delays: TimeSpan,
    last_event_time: TimePoint,

    last_processed_client_span: TimeSpan,
    last_processed_client_time: TimePoint,
}

impl Bank {

    fn update_on_event_end(&mut self, current_t: TimePoint) {
        let delay = current_t - self.last_event_time;
        self.event_delays += delay;
        let mut busy_cashiers_count = BusyCashiersCount::default();
        for c in &mut self.cashiers.iter_mut() {
            c.queue_time_span += c.queue_size * delay;
            match c.is_busy {
                CashierBusy::Busy => busy_cashiers_count += BusyCashiersCount(1.0),
                CashierBusy::NotBusy => (),
            }
        }
        self.busy_cashiers_count_time_span += busy_cashiers_count * delay;
        self.last_event_time = current_t;
    }

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

#[derive(Debug, Copy, Clone, Default, AddAssign)]
struct RefusedCount(usize);

#[derive(Debug, Copy, Clone, Default, AddAssign)]
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

#[derive(Debug, Clone, Copy, AddAssign, Add, Default)]
struct ClientsCount(u64);

#[derive(Debug, Clone, Copy, Default)]
enum CashierBusy {
    #[default]
    NotBusy,
    Busy,
}

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
    work_time: TimeSpan,
    current_t: TimePoint,
    bank: Rc<RefCell<Bank>>,
    cashier_index: CashierIndex
}

impl EventCreate {
    fn iterate(self) -> (EventCreate, Option<EventProcess>) {
        defer! {
            let mut bank = self.bank.borrow_mut();
            bank.update_on_event_end(self.current_t);

            let create_delay = self.current_t - bank.last_create_event_time;
            bank.create_event_time_span += ClientsCount(1) * create_delay;
            bank.last_create_event_time = self.current_t;
        }
        self.bank.borrow_mut().clients_count += ClientsCount(1);
        let work_time = self.process_delay_gen.sample();
        let new_current_t = self.current_t + work_time;
        let fist_busy = self.bank.borrow().get_cashier(CashierIndex::First).is_busy;
        let second_busy = self.bank.borrow().get_cashier(CashierIndex::Second).is_busy;
        (
            EventCreate {
                current_t: self.current_t + self.create_delay_gen.sample(),
                create_delay_gen: self.create_delay_gen,
                process_delay_gen: self.process_delay_gen,
                bank: self.bank.clone()
            },
            match (fist_busy, second_busy) {
                (CashierBusy::NotBusy, CashierBusy::NotBusy) => {
                    let mut bank = self.bank.borrow_mut();
                    assert_eq!(bank.get_cashier(CashierIndex::First).queue_size, QueueSize(0));
                    assert_eq!(bank.get_cashier(CashierIndex::Second).queue_size, QueueSize(0));
                    bank.get_cashier_mut(CashierIndex::First).is_busy = CashierBusy::Busy;
                    Some(EventProcess {
                        delay_gen: self.process_delay_gen,
                        work_time,
                        current_t: new_current_t,
                        bank: self.bank.clone(),
                        cashier_index: CashierIndex::First
                    })
                },
                (CashierBusy::Busy, CashierBusy::NotBusy) => {
                    let mut bank = self.bank.borrow_mut();
                    assert_eq!(bank.get_cashier(CashierIndex::Second).queue_size, QueueSize(0));
                    bank.get_cashier_mut(CashierIndex::Second).is_busy = CashierBusy::Busy;
                    Some(EventProcess {
                        delay_gen: self.process_delay_gen,
                        work_time,
                        current_t: new_current_t,
                        bank: self.bank.clone(),
                        cashier_index: CashierIndex::Second
                    })
                },
                (CashierBusy::NotBusy, CashierBusy::Busy) => {
                    let mut bank = self.bank.borrow_mut();
                    assert_eq!(bank.get_cashier(CashierIndex::First).queue_size, QueueSize(0));
                    bank.get_cashier_mut(CashierIndex::First).is_busy = CashierBusy::Busy;
                    Some(EventProcess {
                        delay_gen: self.process_delay_gen,
                        work_time,
                        current_t: new_current_t,
                        bank: self.bank.clone(),
                        cashier_index: CashierIndex::First
                    })
                }
                (CashierBusy::Busy, CashierBusy::Busy) => {
                    let cashier_index = {
                        let bank = self.bank.borrow_mut();
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
                    }
                    None
                }
            }
        )
    }
}

impl EventProcess {
    fn iterate(self) -> Option<EventProcess> {
        defer! {
            self.bank.borrow_mut().update_on_event_end(self.current_t);
        }

        let mut bank = self.bank.borrow_mut();
        let queue_size = bank.get_cashier(self.cashier_index).queue_size;
        bank.get_cashier_mut(self.cashier_index).is_busy = CashierBusy::NotBusy;
        if queue_size > QueueSize(0) {
            bank.get_cashier_mut(self.cashier_index).queue_size -= QueueSize(1);

            let res = balance_queues(
                bank.get_cashier(CashierIndex::First).queue_size,
                bank.get_cashier(CashierIndex::Second).queue_size,
            );
            bank.get_cashier_mut(CashierIndex::First).queue_size = res.0;
            bank.get_cashier_mut(CashierIndex::Second).queue_size = res.1;
            bank.balance_count += res.2;

            {
                let cashier = bank.get_cashier_mut(self.cashier_index);
                cashier.is_busy = CashierBusy::Busy;
                cashier.work_time += self.work_time;
                cashier.processed_clients += ClientsCount(1);
            }

            let time_between = bank.last_processed_client_time;
            bank.last_processed_client_span += self.current_t - time_between;
            bank.last_processed_client_time = self.current_t;

            let work_time = self.delay_gen.sample();
            Some(EventProcess {
                delay_gen: self.delay_gen,
                work_time,
                current_t: self.current_t + work_time,
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
    const CREATE_MEAN: f64 = 0.5;
    const PROCESS_MEAN: f64 = 0.3;


    let start_time = TimePoint::default();
    let end_time = TimePoint(1000.0);
    let bank: Rc<RefCell<Bank>> = Default::default();
    let mut nodes = vec![
        Event::EventCreate(EventCreate {
            current_t: start_time,
            create_delay_gen: DelayGen::Exponential(
                Exp::new(CREATE_MEAN).expect("Could not create delay gen")
            ),
            process_delay_gen: DelayGen::Exponential(
                Exp::new(PROCESS_MEAN).expect("Could not create delay gen")
            ),
            bank,
        })
    ];


    let last_event = loop {
        nodes.sort_by(|a, b| {
            b.get_current_t().partial_cmp(&a.get_current_t())
                .expect("Can not compare events current_t")
        });
        let next_event = nodes.pop().unwrap();
        if next_event.get_current_t() > end_time {
            break next_event;
        }
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

    let bank = match last_event {
        Event::EventCreate(event) => event.bank,
        Event::EventProcess(event) => event.bank
    };
    let bank = bank.borrow();
    let cashier_first = bank.get_cashier(CashierIndex::First);
    let cashier_second = bank.get_cashier(CashierIndex::Second);
    let total_processed_clients = cashier_first.processed_clients + cashier_second.processed_clients;

    let input_rate = bank.clients_count.0 as f64 / bank.event_delays.0 as f64;

    let time_between_lefts = bank.last_processed_client_span.0 as f64 / total_processed_clients.0 as f64;
    let output_rate = 1.0 / time_between_lefts;

    let utilization = output_rate / input_rate;
    println!("1) utilization: {:?}", utilization);
    println!("1) theoretical utilization: {}", PROCESS_MEAN / CREATE_MEAN);

    println!("2) average_clients_in_bank: {:?}", bank.busy_cashiers_count_time_span / bank.event_delays);
    println!("3) time_between_lefts: {:?}", time_between_lefts);
    println!(
        "4) mean_client_time: {:?}",
        (cashier_first.work_time + cashier_second.work_time).0 as f64
        / total_processed_clients.0 as f64
    );
    println!("5) first_mean_clients_in_queue: {:?}", cashier_first.queue_time_span / bank.event_delays);
    println!("5) second_mean_clients_in_queue: {:?}", cashier_second.queue_time_span / bank.event_delays);
    println!("6) refused_count: {:?}", bank.refused_count.0 as f64 / bank.clients_count.0 as f64);
    println!("7) balance_count: {:?}", bank.balance_count);
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_bank() {

    }
}


