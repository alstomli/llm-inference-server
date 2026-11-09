package main

import (
	"context"
	"errors"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/alstomli/llm-inference-server/gateway"
)

func run() error {
	cfg, err := gateway.ConfigFromEnv()
	if err != nil {
		return err
	}
	handler, err := gateway.New(cfg)
	if err != nil {
		return err
	}
	defer handler.Close()
	host, exists := os.LookupEnv("CHAT_HOST")
	if !exists {
		host = "127.0.0.1"
	}
	port, exists := os.LookupEnv("CHAT_PORT")
	if !exists {
		port = "8001"
	}
	n, err := strconv.Atoi(port)
	if err != nil || n < 1 || n > 65535 {
		return errors.New("invalid CHAT_PORT")
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	server := &http.Server{Addr: net.JoinHostPort(host, port), Handler: handler, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 10 * time.Second, IdleTimeout: 60 * time.Second, MaxHeaderBytes: 16 * 1024, BaseContext: func(net.Listener) context.Context { return ctx }}
	listener, err := net.Listen("tcp", server.Addr)
	if err != nil {
		return err
	}
	log.Print("chat gateway listening on ", server.Addr)
	return serve(ctx, server, listener)
}

// Wait for cancellation cleanup before returning to main. Shutdown closes the
// listener before waiting for handlers, so Serve returning alone is insufficient.
func serve(ctx context.Context, server *http.Server, listener net.Listener) error {
	shutdownDone := make(chan struct{})
	serveDone := make(chan struct{})
	go func() {
		defer close(shutdownDone)
		select {
		case <-ctx.Done():
		case <-serveDone:
			return
		}
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if server.Shutdown(shutdownCtx) != nil {
			_ = server.Close()
		}
	}()
	err := server.Serve(listener)
	close(serveDone)
	<-shutdownDone
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}
func main() {
	if err := run(); err != nil {
		log.Print("chat gateway: ", err)
		os.Exit(1)
	}
}
