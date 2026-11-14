package gateway

import (
	"errors"
	"math"
	"sync"
	"time"
)

// Admission holds a reservation through preparation and the entire response.
// All account and global decisions happen under one lock, including promotion.
type Admission struct {
	mu                                             sync.Mutex
	concurrency, queueLimit, rateLimit             int
	userConcurrency, userQueueLimit, userRateLimit int
	active                                         int
	waiting                                        []*ticket
	accounts                                       map[string]*account
	recent                                         []time.Time
	clock                                          func() time.Time
}
type ticket struct {
	user             string
	ready            chan struct{}
	active, released bool
}
type account struct {
	active, waiting int
	recent          []time.Time
}

func newAdmission(c, q, r, uc, uq, ur int) (*Admission, error) {
	if c < 1 || c > 8 || q < 0 || q > 24 || r < 1 || r > 80 || uc < 1 || uc > c || uq < 0 || uq > q || ur < 1 || ur > r {
		return nil, errors.New("invalid admission limits")
	}
	return &Admission{concurrency: c, queueLimit: q, rateLimit: r, userConcurrency: uc, userQueueLimit: uq, userRateLimit: ur, accounts: make(map[string]*account), clock: time.Now}, nil
}
func pruneTimes(v []time.Time, now time.Time) []time.Time {
	i := 0
	for i < len(v) && !v[i].After(now.Add(-time.Minute)) {
		i++
	}
	return v[i:]
}
func (a *Admission) prune(now time.Time) {
	a.recent = pruneTimes(a.recent, now)
	for u, s := range a.accounts {
		s.recent = pruneTimes(s.recent, now)
		if s.active == 0 && s.waiting == 0 && len(s.recent) == 0 {
			delete(a.accounts, u)
		}
	}
}
func (a *Admission) promote() {
	for i := 0; i < len(a.waiting) && a.active < a.concurrency; {
		t := a.waiting[i]
		s := a.accounts[t.user]
		if s.active >= a.userConcurrency {
			i++
			continue
		}
		a.waiting = append(a.waiting[:i], a.waiting[i+1:]...)
		s.waiting--
		s.active++
		a.active++
		t.active = true
		close(t.ready)
	}
}
func retryAfter(first, now time.Time) int {
	n := int(math.Ceil(first.Add(time.Minute).Sub(now).Seconds()))
	if n < 1 {
		return 1
	}
	if n > 60 {
		return 60
	}
	return n
}
func (a *Admission) reserve(user string) (*ticket, string, int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	now := a.clock()
	a.prune(now)
	a.promote()
	s := a.accounts[user]
	if len(a.recent) >= a.rateLimit {
		return nil, "rate_limit", retryAfter(a.recent[0], now)
	}
	if s != nil && len(s.recent) >= a.userRateLimit {
		return nil, "user_rate_limit", retryAfter(s.recent[0], now)
	}
	start := a.active < a.concurrency && (s == nil || s.active < a.userConcurrency)
	if !start {
		if a.userQueueLimit == 0 || (s != nil && s.waiting >= a.userQueueLimit) {
			return nil, "user_capacity", 0
		}
		if len(a.waiting) >= a.queueLimit {
			return nil, "queue_full", 0
		}
	}
	if s == nil {
		s = &account{}
		a.accounts[user] = s
	}
	t := &ticket{user: user, ready: make(chan struct{}), active: start}
	if start {
		a.active++
		s.active++
		close(t.ready)
	} else {
		s.waiting++
		a.waiting = append(a.waiting, t)
	}
	a.recent = append(a.recent, now)
	s.recent = append(s.recent, now)
	return t, "", 0
}
func (a *Admission) release(t *ticket) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if t == nil || t.released {
		return
	}
	t.released = true
	s := a.accounts[t.user]
	if t.active {
		a.active--
		s.active--
	} else {
		for i, w := range a.waiting {
			if w == t {
				a.waiting = append(a.waiting[:i], a.waiting[i+1:]...)
				s.waiting--
				break
			}
		}
	}
	a.promote()
	a.prune(a.clock())
}
func (a *Admission) snapshot() map[string]int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return map[string]int{"activeRequests": a.active, "waitingRequests": len(a.waiting), "maxConcurrentRequests": a.concurrency, "maxWaitingRequests": a.queueLimit}
}
