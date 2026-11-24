// Command bourbon runs a one-shot lookup comparison of baseline WiscKey-
// style lookups versus Bourbon's learned-model path (paper section 5).
package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"bourbon/bench"
	"bourbon/learn"
)

func main() {
	dataset := flag.String("dataset", "linear", "dataset: linear|seg1|seg10|normal|random")
	nKeys := flag.Int("keys", 200000, "number of keys to load")
	nOps := flag.Int("ops", 100000, "number of lookup operations")
	modeStr := flag.String("mode", "always", "learn mode: always|cba|never")
	flag.Parse()

	var mode learn.Mode
	switch *modeStr {
	case "always":
		mode = learn.ModeAlways
	case "cba":
		mode = learn.ModeCostBenefit
	case "never":
		mode = learn.ModeNever
	default:
		fmt.Fprintf(os.Stderr, "unknown mode %q\n", *modeStr)
		os.Exit(2)
	}

	fmt.Printf("dataset=%s keys=%d ops=%d mode=%s\n",
		*dataset, *nKeys, *nOps, mode)

	start := time.Now()
	res, err := bench.Compare(*dataset, *nKeys, *nOps, mode)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	elapsed := time.Since(start)

	fmt.Printf("baseline avg: %s\n", res.Baseline)
	fmt.Printf("model    avg: %s\n", res.Model)
	fmt.Printf("speedup:     %.2fx\n", res.Speedup)
	fmt.Printf("learned:     %d  skipped: %d  model hits: %d\n",
		res.Learned, res.Skipped, res.ModelHits)
	fmt.Printf("wall time:   %s\n", elapsed)
}
