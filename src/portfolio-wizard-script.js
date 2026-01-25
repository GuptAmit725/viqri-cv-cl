// Portfolio Wizard Script

// Configuration
const API_BASE_URL = 'https://viqri-cv-api-5u7hdc64va-uc.a.run.app';  // Update with your API URL

// State Management
let currentStep = 1;
let wizardData = {
    githubToken: '',
    githubUsername: '',
    repoName: '',
    cvData: null,
    userInfo: null
};

// Initialize
init();

function init() {
    console.log('🚀 Portfolio Wizard Initialized');
    
    // Load CV data from localStorage
    loadCVData();
    
    // Setup event listeners
    setupEventListeners();
    
    // Show first step
    showStep(1);
}

function setupEventListeners() {
    // Token input live update
    const tokenInput = document.getElementById('githubToken');
    if (tokenInput) {
        tokenInput.addEventListener('input', () => {
            wizardData.githubToken = tokenInput.value.trim();
        });
    }
    
    // Repo name input live update
    const repoInput = document.getElementById('repoName');
    if (repoInput) {
        repoInput.addEventListener('input', (e) => {
            wizardData.repoName = e.target.value.trim();
            updatePortfolioUrlPreview();
        });
    }
    
    // Enter key handlers
    document.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const activeStep = document.querySelector('.wizard-step.active');
            if (activeStep) {
                const stepId = activeStep.id;
                if (stepId === 'step2') {
                    verifyToken();
                } else if (stepId === 'step3') {
                    nextStep();
                }
            }
        }
    });
}

function loadCVData() {
    try {
        const cvData = JSON.parse(localStorage.getItem('cvData') || '{}');
        
        if (!cvData || Object.keys(cvData).length === 0) {
            console.warn('No CV data found in localStorage');
            showError('No CV data found. Please upload a CV first.');
            setTimeout(() => {
                window.location.href = 'index.html';
            }, 3000);
            return;
        }
        
        wizardData.cvData = cvData;
        console.log('✅ CV data loaded:', cvData);
    } catch (error) {
        console.error('Error loading CV data:', error);
        showError('Error loading CV data');
    }
}

// Step Navigation
function showStep(step) {
    // Hide all steps
    document.querySelectorAll('.wizard-step').forEach(s => {
        s.classList.remove('active');
    });
    
    // Show target step
    const targetStep = document.getElementById(`step${step}`);
    if (targetStep) {
        targetStep.classList.add('active');
    }
    
    // Update progress tracker
    document.querySelectorAll('.progress-step').forEach((s, index) => {
        if (index + 1 < step) {
            s.classList.add('completed');
            s.classList.remove('active');
        } else if (index + 1 === step) {
            s.classList.add('active');
            s.classList.remove('completed');
        } else {
            s.classList.remove('active', 'completed');
        }
    });
    
    currentStep = step;
    
    // Update summary on step 4
    if (step === 4) {
        updateDeploymentSummary();
    }
}

function nextStep() {
    if (validateCurrentStep()) {
        showStep(currentStep + 1);
    }
}

function prevStep() {
    if (currentStep > 1) {
        showStep(currentStep - 1);
    }
}

function validateCurrentStep() {
    if (currentStep === 2) {
        if (!wizardData.githubToken) {
            showError('Please enter a GitHub token');
            return false;
        }
        if (!wizardData.userInfo) {
            showError('Please verify your GitHub token first');
            return false;
        }
    }
    
    if (currentStep === 3) {
        if (!wizardData.repoName) {
            showError('Please enter a repository name');
            return false;
        }
    }
    
    return true;
}

// GitHub Token Verification
async function verifyToken() {
    const token = document.getElementById('githubToken').value.trim();
    
    if (!token) {
        showError('Please enter a GitHub token');
        return;
    }
    
    const verifyBtn = document.getElementById('verifyTokenBtn');
    const resultDiv = document.getElementById('tokenVerificationResult');
    
    // Show loading
    verifyBtn.disabled = true;
    verifyBtn.innerHTML = `
        <div class="spinner" style="width: 16px; height: 16px; border-width: 2px; margin: 0 auto;"></div>
        Verifying...
    `;
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/verify-github-token`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                github_token: token
            })
        });
        
        const result = await response.json();
        
        if (result.success && result.data.success) {
            // Store user info
            wizardData.githubToken = token;
            wizardData.githubUsername = result.data.username;
            wizardData.userInfo = result.data;
            
            // Show success message
            resultDiv.innerHTML = `
                <div class="info-box success" style="margin-top: 1rem;">
                    <h3>✅ Token Verified Successfully!</h3>
                    <p><strong>Username:</strong> ${result.data.username}</p>
                    ${result.data.name ? `<p><strong>Name:</strong> ${result.data.name}</p>` : ''}
                    ${result.data.email ? `<p><strong>Email:</strong> ${result.data.email}</p>` : ''}
                </div>
            `;
            
            // Update button to "Continue"
            verifyBtn.innerHTML = `
                Continue
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="5" y1="12" x2="19" y2="12"/>
                    <polyline points="12 5 19 12 12 19"/>
                </svg>
            `;
            verifyBtn.onclick = nextStep;
            verifyBtn.disabled = false;
            
        } else {
            throw new Error(result.error || 'Token verification failed');
        }
        
    } catch (error) {
        console.error('Token verification error:', error);
        resultDiv.innerHTML = `
            <div class="warning-box" style="margin-top: 1rem;">
                <strong>❌ Verification Failed:</strong> ${error.message}
            </div>
        `;
        
        // Reset button
        verifyBtn.innerHTML = `
            Verify Token
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="20 6 9 17 4 12"/>
            </svg>
        `;
        verifyBtn.disabled = false;
    }
}

// Toggle Token Visibility
function toggleTokenVisibility() {
    const tokenInput = document.getElementById('githubToken');
    const toggleText = document.getElementById('toggleTokenText');
    
    if (tokenInput.type === 'password') {
        tokenInput.type = 'text';
        toggleText.textContent = 'Hide token';
    } else {
        tokenInput.type = 'password';
        toggleText.textContent = 'Show token';
    }
}

// Accordion Toggle
function toggleAccordion(header) {
    const accordion = header.parentElement;
    accordion.classList.toggle('open');
}

// Update Repository Configuration
function updateRepoConfig() {
    const username = wizardData.githubUsername;
    
    if (!username) return;
    
    // Show user info
    document.getElementById('userInfoDisplay').innerHTML = `
        <h3>👤 GitHub Account</h3>
        <p><strong>Username:</strong> ${username}</p>
        ${wizardData.userInfo.name ? `<p><strong>Name:</strong> ${wizardData.userInfo.name}</p>` : ''}
    `;
    
    // Set default repo name
    const defaultRepo = `${username}.github.io`;
    const repoInput = document.getElementById('repoName');
    repoInput.value = defaultRepo;
    repoInput.placeholder = defaultRepo;
    wizardData.repoName = defaultRepo;
    
    // Update recommended name
    document.getElementById('recommendedRepoName').textContent = username;
    
    // Update URL preview
    updatePortfolioUrlPreview();
}

function updatePortfolioUrlPreview() {
    const username = wizardData.githubUsername;
    const repoName = wizardData.repoName || `${username}.github.io`;
    
    let portfolioUrl = `https://${username}.github.io`;
    
    // If repo name is not username.github.io, add repo name to URL
    if (repoName !== `${username}.github.io`) {
        portfolioUrl = `https://${username}.github.io/${repoName}`;
    }
    
    document.getElementById('portfolioUrlPreview').textContent = portfolioUrl;
}

// Update Deployment Summary
function updateDeploymentSummary() {
    // Update user info in step 3 if not already done
    if (!document.getElementById('userInfoDisplay').innerHTML) {
        updateRepoConfig();
    }
    
    const username = wizardData.githubUsername;
    const repoName = wizardData.repoName;
    
    let portfolioUrl = `https://${username}.github.io`;
    if (repoName !== `${username}.github.io`) {
        portfolioUrl = `https://${username}.github.io/${repoName}`;
    }
    
    document.getElementById('summaryUsername').textContent = username;
    document.getElementById('summaryRepo').textContent = repoName;
    document.getElementById('summaryUrl').textContent = portfolioUrl;
}

// Preview Portfolio
async function previewPortfolio() {
    const modal = document.getElementById('previewModal');
    const frame = document.getElementById('previewFrame');
    
    try {
        // Show loading
        frame.srcdoc = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; font-family: sans-serif;">Loading preview...</div>';
        modal.style.display = 'flex';
        
        const response = await fetch(`${API_BASE_URL}/api/preview-portfolio`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(wizardData.cvData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            frame.srcdoc = result.html;
        } else {
            throw new Error(result.error || 'Preview generation failed');
        }
        
    } catch (error) {
        console.error('Preview error:', error);
        frame.srcdoc = `<div style="display: flex; align-items: center; justify-content: center; height: 100%; font-family: sans-serif; color: #f56565;">Error: ${error.message}</div>`;
    }
}

function closePreview() {
    document.getElementById('previewModal').style.display = 'none';
}

// Deploy Portfolio
async function deployPortfolio() {
    const deployBtn = document.getElementById('deployBtn');
    const progressDiv = document.getElementById('deploymentProgress');
    const successDiv = document.getElementById('deploymentSuccess');
    const statusText = document.getElementById('deploymentStatusText');
    
    // Hide button, show progress
    deployBtn.style.display = 'none';
    document.querySelector('.deployment-summary').style.display = 'none';
    document.querySelector('.preview-section').style.display = 'none';
    document.querySelector('.button-group').style.display = 'none';
    progressDiv.style.display = 'block';
    
    try {
        // Step 1: Creating repository
        statusText.textContent = 'Creating GitHub repository...';
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Step 2: Generating portfolio
        statusText.textContent = 'Generating your portfolio...';
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Step 3: Deploying
        statusText.textContent = 'Deploying to GitHub Pages...';
        
        const response = await fetch(`${API_BASE_URL}/api/deploy-portfolio`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                github_token: wizardData.githubToken,
                github_username: wizardData.githubUsername,
                repo_name: wizardData.repoName,
                cv_data: wizardData.cvData
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Show success
            progressDiv.style.display = 'none';
            successDiv.style.display = 'block';
            
            const portfolioUrl = result.data.portfolio_url;
            document.getElementById('successPortfolioUrl').href = portfolioUrl;
            document.getElementById('successPortfolioUrl').textContent = portfolioUrl;
            
            // Celebration effect
            celebrate();
            
        } else {
            throw new Error(result.error || 'Deployment failed');
        }
        
    } catch (error) {
        console.error('Deployment error:', error);
        
        progressDiv.style.display = 'none';
        
        // Show error
        alert(`Deployment failed: ${error.message}`);
        
        // Restore UI
        deployBtn.style.display = 'flex';
        document.querySelector('.deployment-summary').style.display = 'block';
        document.querySelector('.preview-section').style.display = 'block';
        document.querySelector('.button-group').style.display = 'flex';
    }
}

// Utility Functions
function showError(message) {
    alert(message);
}

function celebrate() {
    // Simple celebration - could be enhanced with confetti library
    console.log('🎉 Portfolio deployed successfully!');
}

// Export for debugging
window.wizardDebug = {
    data: wizardData,
    goToStep: showStep,
    currentStep: () => currentStep
};

console.log('✅ Portfolio Wizard Script Loaded');