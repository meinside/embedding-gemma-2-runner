package assets

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Prunable lists what the cache holds for other versions than the pinned
// ones, and what interrupted installs left behind: library directories of
// other llama.cpp releases, staging directories of this one, model files of
// other repositories and revisions. Nothing the current pins use is listed,
// not even a partial download of it, which the next run resumes.
func Prunable(d Dirs) ([]string, error) {
	var out []string
	add := func(p string) { out = append(out, p) }

	// lib/<version>/<os-arch>/<processor>, with <processor>.*.part staging dirs
	versions, err := readDir(d.Libs)
	if err != nil {
		return nil, err
	}
	for _, v := range versions {
		if v.Name() != LlamaVersion {
			add(filepath.Join(d.Libs, v.Name()))
			continue
		}
		platforms, err := readDir(filepath.Join(d.Libs, v.Name()))
		if err != nil {
			return nil, err
		}
		for _, p := range platforms {
			entries, err := readDir(filepath.Join(d.Libs, v.Name(), p.Name()))
			if err != nil {
				return nil, err
			}
			for _, e := range entries {
				if strings.Contains(e.Name(), ".part") {
					add(filepath.Join(d.Libs, v.Name(), p.Name(), e.Name()))
				}
			}
		}
	}

	// models/<owner>/<repo>/<revision>
	owners, err := readDir(d.Models)
	if err != nil {
		return nil, err
	}
	owner, repo, _ := strings.Cut(ModelRepo, "/")
	for _, o := range owners {
		if o.Name() != owner {
			add(filepath.Join(d.Models, o.Name()))
			continue
		}
		repos, err := readDir(filepath.Join(d.Models, o.Name()))
		if err != nil {
			return nil, err
		}
		for _, r := range repos {
			if r.Name() != repo {
				add(filepath.Join(d.Models, o.Name(), r.Name()))
				continue
			}
			revs, err := readDir(filepath.Join(d.Models, o.Name(), r.Name()))
			if err != nil {
				return nil, err
			}
			for _, rev := range revs {
				if rev.Name() != ModelRevision {
					add(filepath.Join(d.Models, o.Name(), r.Name(), rev.Name()))
				}
			}
		}
	}
	sort.Strings(out)
	return out, nil
}

// readDir lists dir, empty for a directory that does not exist.
func readDir(dir string) ([]os.DirEntry, error) {
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil, nil
	}
	return entries, err
}
