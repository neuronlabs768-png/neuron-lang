/// NEURON Package Manager — Resolves and builds dependencies from neuron.toml.

use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Dependency {
    pub name: String,
    pub path: Option<String>,
    pub git: Option<String>,
}

pub fn build_package(project_dir: &str) -> Result<String, String> {
    let project_path = Path::new(project_dir);
    let toml_path = project_path.join("neuron.toml");
    
    if !toml_path.exists() {
        return Err(format!("No neuron.toml found in {}", project_dir));
    }

    println!("Building NEURON package in: {}", project_dir);

    let toml_content = fs::read_to_string(&toml_path)
        .map_err(|e| format!("Failed to read neuron.toml: {}", e))?;

    let (pkg_name, deps) = parse_neuron_toml(&toml_content)?;
    println!("Package: {}", pkg_name);

    // Concatenate source files of dependencies, then the main source files
    let mut combined_source = String::new();

    for dep in deps {
        println!("Resolving dependency: {}", dep.name);
        let dep_src = resolve_dependency(&dep, project_path)?;
        combined_source.push_str(&format!("\n// --- Dependency: {} ---\n", dep.name));
        combined_source.push_str(&dep_src);
        combined_source.push_str("\n");
    }

    // Now append the main source files of this package
    // Usually located in src/main.neuron, src/main.nr, main.neuron, or main.nr
    let src_dir = project_path.join("src");
    let main_nr = if src_dir.join("main.neuron").exists() {
        src_dir.join("main.neuron")
    } else if src_dir.join("main.nr").exists() {
        src_dir.join("main.nr")
    } else if project_path.join("main.neuron").exists() {
        project_path.join("main.neuron")
    } else if project_path.join("main.nr").exists() {
        project_path.join("main.nr")
    } else {
        project_path.join("main.neuron")
    };

    if !main_nr.exists() {
        return Err(format!("Could not find main.neuron or src/main.neuron (nor .nr variants) in {}", project_dir));
    }

    let main_source = fs::read_to_string(&main_nr)
        .map_err(|e| format!("Failed to read source file {}: {}", main_nr.display(), e))?;

    combined_source.push_str(&format!("\n// --- Package: {} ---\n", pkg_name));
    combined_source.push_str(&main_source);

    println!("✓ Resolved and built all dependencies successfully.");
    Ok(combined_source)
}

pub fn add_dependency(project_dir: &str, dep_name: &str, path: Option<&str>, git: Option<&str>) -> Result<(), String> {
    let toml_path = Path::new(project_dir).join("neuron.toml");
    if !toml_path.exists() {
        return Err(format!("No neuron.toml found in {}", project_dir));
    }

    let mut toml_content = fs::read_to_string(&toml_path)
        .map_err(|e| format!("Failed to read neuron.toml: {}", e))?;

    // Append dependency definition to the end
    let dep_str = if let Some(p) = path {
        format!("{} = {{ path = {:?} }}\n", dep_name, p)
    } else if let Some(g) = git {
        format!("{} = {{ git = {:?} }}\n", dep_name, g)
    } else {
        return Err("Must specify either a path or git URL for the dependency".into());
    };

    if !toml_content.contains("[dependencies]") {
        toml_content.push_str("\n[dependencies]\n");
    }

    toml_content.push_str(&dep_str);

    fs::write(&toml_path, toml_content)
        .map_err(|e| format!("Failed to update neuron.toml: {}", e))?;

    println!("Added dependency '{}' to neuron.toml", dep_name);
    Ok(())
}

fn parse_neuron_toml(content: &str) -> Result<(String, Vec<Dependency>), String> {
    let mut pkg_name = "unknown".to_string();
    let mut deps = Vec::new();
    let mut in_deps_section = false;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if line == "[package]" {
            in_deps_section = false;
            continue;
        }

        if line == "[dependencies]" {
            in_deps_section = true;
            continue;
        }

        if line.starts_with('[') {
            in_deps_section = false;
            continue;
        }

        if in_deps_section {
            // Parse: name = { path = "..." } or name = { git = "..." }
            let parts: Vec<&str> = line.splitn(2, '=').collect();
            if parts.len() == 2 {
                let name = parts[0].trim().to_string();
                let val = parts[1].trim();

                let mut path = None;
                let mut git = None;

                if val.contains("path") {
                    if let Some(start) = val.find("path") {
                        let sub = &val[start..];
                        if let Some(q_start) = sub.find('"') {
                            let remain = &sub[q_start + 1..];
                            if let Some(q_end) = remain.find('"') {
                                path = Some(remain[..q_end].to_string());
                            }
                        }
                    }
                } else if val.contains("git") {
                    if let Some(start) = val.find("git") {
                        let sub = &val[start..];
                        if let Some(q_start) = sub.find('"') {
                            let remain = &sub[q_start + 1..];
                            if let Some(q_end) = remain.find('"') {
                                git = Some(remain[..q_end].to_string());
                            }
                        }
                    }
                }

                deps.push(Dependency { name, path, git });
            }
        } else {
            // Parse package name: name = "..."
            if line.starts_with("name") {
                let parts: Vec<&str> = line.splitn(2, '=').collect();
                if parts.len() == 2 {
                    let val = parts[1].trim();
                    if let Some(start) = val.find('"') {
                        let remain = &val[start + 1..];
                        if let Some(end) = remain.find('"') {
                            pkg_name = remain[..end].to_string();
                        }
                    }
                }
            }
        }
    }

    Ok((pkg_name, deps))
}

fn resolve_dependency(dep: &Dependency, project_path: &Path) -> Result<String, String> {
    if let Some(ref path_str) = dep.path {
        let dep_path = project_path.join(path_str);
        let src_dir = dep_path.join("src");
        let main_nr = if src_dir.join("main.neuron").exists() {
            src_dir.join("main.neuron")
        } else if src_dir.join("main.nr").exists() {
            src_dir.join("main.nr")
        } else if dep_path.join("main.neuron").exists() {
            dep_path.join("main.neuron")
        } else if dep_path.join("main.nr").exists() {
            dep_path.join("main.nr")
        } else if dep_path.join(&format!("{}.neuron", dep.name)).exists() {
            dep_path.join(&format!("{}.neuron", dep.name))
        } else {
            dep_path.join(&format!("{}.nr", dep.name))
        };

        if !main_nr.exists() {
            return Err(format!("Could not find main file (main.neuron or main.nr) in dependency '{}' at {:?}", dep.name, dep_path));
        }

        fs::read_to_string(&main_nr)
            .map_err(|e| format!("Failed to read dependency '{}' source: {}", dep.name, e))
    } else if let Some(ref git_url) = dep.git {
        // Mock git resolution by using a cached local dir or downloading
        // In this environment, we simulate cloning a git repo to a local cache directory
        let cache_dir = std::env::temp_dir().join("neuron_git_cache").join(&dep.name);
        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir).map_err(|e| e.to_string())?;
            // Mock a standard library or downloaded library source file
            let mock_src = format!(
                "// Mock downloaded source for git dependency {}\n// Git URL: {}\nfn {}_version() -> String {{ {:?}.to_string() }}\n",
                dep.name, git_url, dep.name, git_url
            );
            fs::write(cache_dir.join("main.nr"), mock_src).map_err(|e| e.to_string())?;
        }
        fs::read_to_string(cache_dir.join("main.nr"))
            .map_err(|e| format!("Failed to read cached git dependency '{}': {}", dep.name, e))
    } else {
        Err(format!("Dependency '{}' must specify either path or git", dep.name))
    }
}
