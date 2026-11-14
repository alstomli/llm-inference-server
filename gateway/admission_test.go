package gateway

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestAdmissionFairPromotionAndCancellation(t *testing.T) {
	a, _ := newAdmission(2, 6, 20, 1, 1, 6)
	a1, _, _ := a.reserve("a")
	a2, _, _ := a.reserve("a")
	b1, _, _ := a.reserve("b")
	c1, _, _ := a.reserve("c")
	if _, why, _ := a.reserve("a"); why != "user_capacity" {
		t.Fatalf("same-account capacity: %q", why)
	}
	a.release(b1)
	select {
	case <-c1.ready:
	default:
		t.Fatal("eligible peer blocked behind busy account")
	}
	select {
	case <-a2.ready:
		t.Fatal("same-account waiter promoted too soon")
	default:
	}
	a.release(a2)
	a.release(a2)
	a.release(a1)
	a.release(c1)
	if got := a.snapshot(); got["activeRequests"] != 0 || got["waitingRequests"] != 0 {
		t.Fatalf("reservation leak: %v", got)
	}
}
func TestAdmissionRollingLimitsAndPruning(t *testing.T) {
	a, _ := newAdmission(2, 6, 20, 1, 1, 2)
	now := time.Now()
	a.clock = func() time.Time { return now }
	for i := 0; i < 2; i++ {
		ticket, _, _ := a.reserve("a")
		a.release(ticket)
	}
	if _, why, retry := a.reserve("a"); why != "user_rate_limit" || retry != 60 {
		t.Fatalf("rate limit: %s %d", why, retry)
	}
	now = now.Add(time.Minute)
	ticket, why, _ := a.reserve("a")
	if why != "" {
		t.Fatal(why)
	}
	a.release(ticket)
	now = now.Add(time.Minute)
	a.mu.Lock()
	a.prune(now)
	if len(a.accounts) != 0 {
		t.Fatal("expired account retained")
	}
	a.mu.Unlock()
}
func TestAdmissionConcurrentReservations(t *testing.T) {
	a, _ := newAdmission(2, 6, 20, 1, 1, 6)
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ticket, _, _ := a.reserve("a")
			if ticket != nil {
				a.release(ticket)
			}
		}()
	}
	wg.Wait()
	if got := a.snapshot(); got["activeRequests"] != 0 || got["waitingRequests"] != 0 {
		t.Fatal(got)
	}
}

func TestAdmissionPromotionCancellationRace(t *testing.T) {
	for run := 0; run < 100; run++ {
		a, _ := newAdmission(2, 6, 20, 1, 1, 6)
		tickets := make([]*ticket, 0, 8)
		for i := 0; i < 8; i++ {
			ticket, why, _ := a.reserve(fmt.Sprintf("account-%d", i))
			if ticket == nil {
				t.Fatal(why)
			}
			tickets = append(tickets, ticket)
		}
		var wg sync.WaitGroup
		for _, reservation := range tickets {
			wg.Add(1)
			go func(ticket *ticket) { defer wg.Done(); a.release(ticket); a.release(ticket) }(reservation)
		}
		wg.Wait()
		snapshot := a.snapshot()
		if snapshot["activeRequests"] != 0 || snapshot["waitingRequests"] != 0 {
			t.Fatalf("iteration %d: %v", run, snapshot)
		}
	}
}

func TestEightActiveTwentyFourWaitingFairnessAndRecovery(t *testing.T) {
	a, err := newAdmission(8, 24, 80, 1, 1, 6)
	if err != nil {
		t.Fatal(err)
	}
	active := make([]*ticket, 8)
	for i := range active {
		active[i], _, _ = a.reserve(fmt.Sprintf("active-%d", i))
	}
	blocked, _, _ := a.reserve("active-0")
	waiters := make([]*ticket, 23)
	for i := range waiters {
		waiters[i], _, _ = a.reserve(fmt.Sprintf("waiter-%d", i))
	}
	if got := a.snapshot(); got["activeRequests"] != 8 || got["waitingRequests"] != 24 {
		t.Fatal(got)
	}
	if _, why, _ := a.reserve("overflow"); why != "queue_full" {
		t.Fatal(why)
	}
	if _, why, _ := a.reserve("active-0"); why != "user_capacity" {
		t.Fatal(why)
	}
	a.release(active[1])
	select {
	case <-waiters[0].ready:
	default:
		t.Fatal("oldest eligible peer not promoted")
	}
	select {
	case <-blocked.ready:
		t.Fatal("busy-account waiter bypassed per-account cap")
	default:
	}
	a.release(active[0])
	select {
	case <-blocked.ready:
	default:
		t.Fatal("oldest now-eligible account not promoted")
	}
	// Queued cancellation and active cleanup release every reservation exactly once.
	for _, v := range waiters {
		a.release(v)
		a.release(v)
	}
	a.release(blocked)
	for _, v := range active {
		a.release(v)
	}
	if got := a.snapshot(); got["activeRequests"] != 0 || got["waitingRequests"] != 0 {
		t.Fatal(got)
	}
	next, why, _ := a.reserve("recovered")
	if next == nil {
		t.Fatal(why)
	}
	a.release(next)
}
func TestEightyGlobalRequestsKeepSixPerAccount(t *testing.T) {
	a, _ := newAdmission(8, 24, 80, 1, 1, 6)
	now := time.Now()
	a.clock = func() time.Time { return now }
	for i := 0; i < 6; i++ {
		v, why, _ := a.reserve("same")
		if v == nil {
			t.Fatal(why)
		}
		a.release(v)
	}
	if _, why, _ := a.reserve("same"); why != "user_rate_limit" {
		t.Fatal(why)
	}
	for i := 6; i < 80; i++ {
		v, why, _ := a.reserve(fmt.Sprintf("user-%d", i))
		if v == nil {
			t.Fatal(why)
		}
		a.release(v)
	}
	if _, why, retry := a.reserve("overflow"); why != "rate_limit" || retry != 60 {
		t.Fatalf("%s %d", why, retry)
	}
	now = now.Add(time.Minute)
	v, why, _ := a.reserve("same")
	if v == nil {
		t.Fatal(why)
	}
	a.release(v)
}
