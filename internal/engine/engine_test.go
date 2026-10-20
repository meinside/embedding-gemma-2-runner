package engine

import (
	"reflect"
	"testing"
)

func TestGroups(t *testing.T) {
	tests := []struct {
		lens              []int
		maxSeq, maxTokens int
		want              [][]int
	}{
		{nil, 4, 100, nil},
		{[]int{10, 20, 30}, 1, 100, [][]int{{0}, {1}, {2}}},
		{[]int{10, 20, 30}, 4, 100, [][]int{{0, 1, 2}}},
		{[]int{10, 20, 30}, 2, 100, [][]int{{0, 1}, {2}}},
		{[]int{60, 50, 40, 10}, 4, 100, [][]int{{0}, {1, 2, 3}}},
		{[]int{10, -1, 20, -1}, 4, 100, [][]int{{0, 2}}}, // too long ones are left out
		{[]int{100, 100}, 4, 100, [][]int{{0}, {1}}},     // each exactly fills a batch
	}
	for _, tt := range tests {
		if got := groups(tt.lens, tt.maxSeq, tt.maxTokens); !reflect.DeepEqual(got, tt.want) {
			t.Errorf("groups(%v, %d, %d) = %v, want %v", tt.lens, tt.maxSeq, tt.maxTokens, got, tt.want)
		}
	}
}

func TestDefaultSeqs(t *testing.T) {
	for _, c := range []struct {
		n    int
		gpu  bool
		want int
	}{{0, true, 32}, {0, false, 1}, {1, true, 1}, {4, false, 4}, {-1, true, 32}} {
		if got := defaultSeqs(c.n, c.gpu); got != c.want {
			t.Errorf("defaultSeqs(%d, %v) = %d, want %d", c.n, c.gpu, got, c.want)
		}
	}
}
