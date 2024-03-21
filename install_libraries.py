import subprocess

required_libraries = ['nltk', 'pandas','transformers', 'pytorch']

def install_libraries():
    for library in required_libraries:
        try:
            subprocess.check_call(['pip', 'install', library])
        except subprocess.CalledProcessError as e:
            print(f"Error installing {library}: {e}")

if __name__ == '__main__':
    install_libraries()