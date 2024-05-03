import subprocess
import yaml

def get_installed_libraries():
    # Get a list of installed packages and their versions
    result = subprocess.run(['pip', 'list', '--format=json'], capture_output=True, text=True)
    installed_packages = result.stdout.strip()
    packages_json = yaml.safe_load(installed_packages)
    return {package['name']: package['version'] for package in packages_json}

def save_to_yaml(data, filename):
    with open(filename, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

if __name__ == "__main__":
    installed_libraries = get_installed_libraries()
    save_to_yaml({"libraries": installed_libraries}, "installed_libraries.yaml")
