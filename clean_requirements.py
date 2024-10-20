import re

with open('new_requirements.txt', 'r') as infile, open('cleaned_requirements.txt', 'w') as outfile:
    for line in infile:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        package_name = re.split('==|>=|<=|~=|!=', line)[0]
        # Simple check: skip known system packages
        system_packages = ['bzip2', '_libgcc_mutex', 'libstdcxx-ng', 'libgcc-ng', 'tk', 'zlib', 'xz', 'sqlite', 'readline']
        if package_name in system_packages:
            print(f"Skipping system package: {package_name}")
            continue
        outfile.write(line + '\n')
