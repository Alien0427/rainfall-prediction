import os
import requests
import subprocess
from getpass import getpass
import time

def create_github_repo():
    # Get GitHub token
    token = getpass("Enter your GitHub Personal Access Token: ")
    
    # Repository details
    repo_name = "rainfall-prediction"
    description = "A machine learning-based web application for rainfall prediction"
    
    # Create repository
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {
        "name": repo_name,
        "description": description,
        "private": False,
        "auto_init": False,
        "has_issues": True,
        "has_wiki": True,
        "has_projects": True
    }
    
    response = requests.post(
        "https://api.github.com/user/repos",
        headers=headers,
        json=data
    )
    
    if response.status_code == 201:
        print("Repository created successfully!")
        return response.json()["clone_url"], token
    else:
        print(f"Error creating repository: {response.json()}")
        return None, None

def setup_github_pages(repo_name, token):
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Enable GitHub Pages
    data = {
        "source": {
            "branch": "gh-pages",
            "path": "/"
        }
    }
    
    response = requests.post(
        f"https://api.github.com/repos/{repo_name}/pages",
        headers=headers,
        json=data
    )
    
    if response.status_code == 201:
        print("GitHub Pages enabled successfully!")
    else:
        print(f"Error enabling GitHub Pages: {response.json()}")

def setup_branch_protection(repo_name, token):
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {
        "required_status_checks": {
            "strict": True,
            "contexts": ["test"]
        },
        "enforce_admins": True,
        "required_pull_request_reviews": {
            "dismissal_restrictions": {},
            "dismiss_stale_reviews": True,
            "require_code_owner_reviews": True
        },
        "restrictions": None
    }
    
    response = requests.put(
        f"https://api.github.com/repos/{repo_name}/branches/master/protection",
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        print("Branch protection rules set successfully!")
    else:
        print(f"Error setting branch protection: {response.json()}")

def push_to_github(repo_url, token):
    try:
        # Add remote
        subprocess.run(["git", "remote", "add", "origin", repo_url], check=True)
        
        # Create and switch to gh-pages branch
        subprocess.run(["git", "checkout", "-b", "gh-pages"], check=True)
        
        # Create a simple index.html for GitHub Pages
        with open("index.html", "w") as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Rainfall Prediction</title>
                <meta http-equiv="refresh" content="0; url=https://your-app-url.onrender.com" />
            </head>
            <body>
                <p>Redirecting to the application...</p>
            </body>
            </html>
            """)
        
        # Add and commit the index.html
        subprocess.run(["git", "add", "index.html"], check=True)
        subprocess.run(["git", "commit", "-m", "Add GitHub Pages redirect"], check=True)
        
        # Push gh-pages branch
        subprocess.run(["git", "push", "-u", "origin", "gh-pages"], check=True)
        
        # Switch back to master branch
        subprocess.run(["git", "checkout", "master"], check=True)
        
        # Push to GitHub
        subprocess.run(["git", "push", "-u", "origin", "master"], check=True)
        
        print("Code pushed to GitHub successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error pushing to GitHub: {e}")
        return False

if __name__ == "__main__":
    print("Deploying to GitHub...")
    
    # Create repository
    repo_url, token = create_github_repo()
    if repo_url and token:
        # Extract repository name from URL
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        
        # Push code
        if push_to_github(repo_url, token):
            # Setup GitHub Pages
            setup_github_pages(repo_name, token)
            
            # Setup branch protection
            setup_branch_protection(repo_name, token)
            
            print("\nDeployment completed successfully!")
            print(f"Repository URL: https://github.com/{repo_name}")
            print(f"GitHub Pages URL: https://{repo_name}.github.io")
            print("\nNext steps:")
            print("1. Add your OpenWeatherMap API key to GitHub Secrets")
            print("2. Add your Render API key to GitHub Secrets")
            print("3. Update the Render deployment URL in the GitHub Actions workflow") 