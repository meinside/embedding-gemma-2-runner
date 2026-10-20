package assets

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
)

// LlamaVersion is the llama-cpp-builder release whose libraries eg2 loads. It
// must be the one the yzma version in go.mod is written against, and contain
// ggml-org/llama.cpp#30054 (EmbeddingGemma 2 support). See libAssets to upgrade.
const (
	LlamaVersion        = "v0.7.0"
	llamaUpstream       = "b11669"
	llamaManifestSHA256 = "f8fac86089bf92657fd33fa1ea0f2ee147b4c729567a99610c64a150d7ca97d3"
)

// Processor names a llama.cpp build. Every build also has the CPU backend, so a
// GPU build still runs where its driver is missing.
type Processor string

const (
	CPU    Processor = "cpu"
	Metal  Processor = "metal"
	Vulkan Processor = "vulkan"
	CUDA12 Processor = "cuda-12"
	CUDA13 Processor = "cuda-13"
)

// libAssets lists, per GOOS/GOARCH and processor, the archives to unpack into
// one directory. Pinned here rather than resolved by yzma's pkg/download,
// which pulls in go-getter and cloud SDKs and grew the eg2 binary from 11 MB
// to 57 MB; asset choices follow yzma's resolver.
//
// URLs and digests come from the llama-cpp-builder v0.7.0.json manifest, whose
// sha256 is llamaManifestSHA256, the DefaultVersion pin in yzma v1.30.0. To
// upgrade, take the new pin from yzma's pkg/download/version.go, check that
// release's manifest against it, and copy the digests here. TestLlamaPinMatchesYzma
// fails when the pins drift apart.
var libAssets = map[string]map[Processor][]pinned{
	"darwin/arm64": {
		Metal: {upstream("llama-b11669-bin-macos-arm64.tar.gz", "dbed55493d72bc1ba2babc8770d669c2289b5bc4ee9fb2a4dfc158d45b7d13fd")},
	},
	"darwin/amd64": {
		CPU: {upstream("llama-b11669-bin-macos-x64.tar.gz", "cab28238d78158109b48deb27b05d50e0dcef56275d2e76da9ca8281526c7532")},
	},
	"linux/amd64": {
		CPU:    {upstream("llama-b11669-bin-ubuntu-x64.tar.gz", "cf07c1c112d709bd67e522017471de57ed0ec8bea031d732c5422351c07f65ac")},
		Vulkan: {upstream("llama-b11669-bin-ubuntu-vulkan-x64.tar.gz", "e8709e5ba6c1b093dea7f3de57f4aba842f9a188fba6b465782d65d6fcfa4d9f")},
		CUDA12: {builder("llama-v0.7.0-bin-ubuntu-cuda-x64.tar.gz", "7cf942787ca2addfdd3b5f68fb2c21351420a89575c80b150eb887c0dfdecabe")},
		CUDA13: {builder("llama-v0.7.0-bin-ubuntu-cuda-13-x64.tar.gz", "5621d0226e389497358bef3294b9c0acf820e1f924acffe1116343c1dcc87acb")},
	},
	"linux/arm64": {
		CPU:    {builder("llama-v0.7.0-bin-ubuntu-cpu-arm64.tar.gz", "ddc92e760022472ee4a4270ca75e211a289d56d836145cc5650dca1771d8563e")},
		Vulkan: {builder("llama-v0.7.0-bin-ubuntu-vulkan-arm64.tar.gz", "ea2072d5357b95bf6aeb282d616b99a5c90dea06fad5cd610f3fef8ddb96ca4c")},
		CUDA12: {builder("llama-v0.7.0-bin-ubuntu-cuda-arm64.tar.gz", "e06fc416b310bc9f1f1f208e37fd1c1a5e3ab94dd8679495ce84eb2dde49531d")},
		CUDA13: {builder("llama-v0.7.0-bin-ubuntu-cuda-13-arm64.tar.gz", "f569c9018dfa45ca7277397f019c0455aff4969540f5bb330e103c226f4bbcb9")},
	},
	"windows/amd64": {
		CPU:    {upstream("llama-b11669-bin-win-cpu-x64.zip", "37b3efcf962847ca869745fd4ccf7f5e7e8dacfeff9fa3523b4ca8cdc66d632b")},
		Vulkan: {upstream("llama-b11669-bin-win-vulkan-x64.zip", "2c07574e7bde71321cb785282b20cc821e951132149eeceeea32ddebdbf0598e")},
		CUDA12: {
			upstream("llama-b11669-bin-win-cuda-12.4-x64.zip", "c2da1f73ed176ae093ac5aceb978a7b167a16113f6d74c6789eeaee2135cb8dc"),
			upstream("cudart-llama-bin-win-cuda-12.4-x64.zip", "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"),
		},
		CUDA13: {
			upstream("llama-b11669-bin-win-cuda-13.4-x64.zip", "3b92d5c064f5032e603e531d3ec8961f3bd6613e5dfad047399a92dec8e95dbc"),
			upstream("cudart-llama-bin-win-cuda-13.4-x64.zip", "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668"),
		},
	},
	"windows/arm64": {
		CPU:    {upstream("llama-b11669-bin-win-cpu-arm64.zip", "9cff39ffed17edcd03a7d876073f74d7b48cd43a1e5eafdd59ee094185ad871b")},
		Vulkan: {upstream("llama-b11669-bin-win-vulkan-arm64.zip", "963d3b185362cadca3de643409f916a5ce95dd0f7fd8617ba9ee5d8833ac1ef3")},
	},
}

func upstream(name, sha string) pinned {
	return pinned{Name: name, URL: "https://github.com/ggml-org/llama.cpp/releases/download/" + llamaUpstream + "/" + name, SHA256: sha}
}

func builder(name, sha string) pinned {
	return pinned{Name: name, URL: "https://github.com/hybridgroup/llama-cpp-builder/releases/download/" + LlamaVersion + "/" + name, SHA256: sha}
}

func platform() string { return runtime.GOOS + "/" + runtime.GOARCH }

// Processors lists the builds available for this platform.
func Processors() []Processor {
	var out []Processor
	for _, p := range []Processor{CPU, Metal, Vulkan, CUDA12, CUDA13} {
		if _, ok := libAssets[platform()][p]; ok {
			out = append(out, p)
		}
	}
	return out
}

// DefaultProcessor picks the build for this machine: Metal on Apple silicon,
// CUDA where nvidia-smi reports a driver, Vulkan on other GPUs, CPU otherwise.
// With gpu false it picks the smallest build that runs on the CPU.
func DefaultProcessor(gpu bool) Processor {
	builds := libAssets[platform()]
	if _, ok := builds[Metal]; ok {
		return Metal // one archive for CPU and GPU
	}
	if !gpu {
		return CPU
	}
	if major := cudaMajor(); major > 0 {
		if _, ok := builds[CUDA13]; ok && major >= 13 {
			return CUDA13
		}
		if _, ok := builds[CUDA12]; ok {
			return CUDA12
		}
	}
	if _, ok := builds[Vulkan]; ok {
		return Vulkan
	}
	return CPU
}

// cudaMajor reports the CUDA version the NVIDIA driver supports, or 0.
func cudaMajor() int {
	out, err := exec.Command("nvidia-smi").Output()
	if err != nil {
		return 0
	}
	m := regexp.MustCompile(`CUDA Version:\s*(\d+)`).FindSubmatch(out)
	if m == nil {
		return 12 // a driver without the banner is old
	}
	n, _ := strconv.Atoi(string(bytes.TrimSpace(m[1])))
	return n
}

// LibOptions controls how the llama.cpp libraries are obtained.
type LibOptions struct {
	Processor Processor // "" picks DefaultProcessor(true)
	Offline   bool
	Progress  Progress
}

// LibDir returns where the libraries for p are, or would be, installed.
func LibDir(d Dirs, p Processor) string {
	return filepath.Join(d.Libs, LlamaVersion, runtime.GOOS+"-"+runtime.GOARCH, string(p))
}

// Libraries returns the directory with the llama.cpp libraries for o.Processor,
// downloading and unpacking them first if needed.
func Libraries(ctx context.Context, d Dirs, o LibOptions) (string, Processor, error) {
	p := o.Processor
	if p == "" {
		p = DefaultProcessor(true)
	}
	files, ok := libAssets[platform()][p]
	if !ok {
		return "", p, fmt.Errorf("no %s build of llama.cpp for %s", p, platform())
	}
	dir := LibDir(d, p)
	if _, err := os.Stat(filepath.Join(dir, completeMarker)); err == nil {
		return dir, p, nil
	}
	if o.Offline {
		return "", p, fmt.Errorf("llama.cpp %s %s libraries: %w", LlamaVersion, p, ErrOffline)
	}

	// Unpack into a sibling directory and rename it into place, so an
	// interrupted install never leaves a directory that looks complete.
	if err := os.MkdirAll(filepath.Dir(dir), 0o755); err != nil {
		return "", p, err
	}
	staging, err := os.MkdirTemp(filepath.Dir(dir), string(p)+".*.part")
	if err != nil {
		return "", p, err
	}
	defer os.RemoveAll(staging)
	for _, f := range files {
		archive, err := fetchTo(ctx, staging, f, o.Progress)
		if err != nil {
			return "", p, err
		}
		if err := unpack(archive, staging); err != nil {
			return "", p, fmt.Errorf("unpack %s: %w", f.Name, err)
		}
		if err := os.Remove(archive); err != nil {
			return "", p, err
		}
	}
	if err := os.WriteFile(filepath.Join(staging, completeMarker), nil, 0o644); err != nil {
		return "", p, err
	}
	if err := os.RemoveAll(dir); err != nil && !errors.Is(err, os.ErrNotExist) {
		return "", p, err
	}
	if err := os.Rename(staging, dir); err != nil {
		return "", p, err
	}
	return dir, p, nil
}

const completeMarker = ".eg2-complete"
