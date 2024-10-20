with open('requirements.txt', 'r') as infile, open('new_requirements.txt', 'w') as outfile:
    for line in infile:
        if line.strip() and not line.startswith('#'):
            parts = line.strip().split('=')
            package = parts[0]
            version = parts[1] if len(parts) > 1 else ''
            outfile.write(f"{package}=={version}\n")
