// Portfolio Wizard Script

// Configuration
const API_BASE_URL = 'https://viqri-cv-api-5u7hdc64va-uc.a.run.app';  // Update with your API URL

// ============================================
// Feedback & Rating  (writes go to Firestore via the backend)
// ============================================
const Feedback = {
  _resolve: null,
  _selected: 0,

  /** Show the modal; returns a promise that resolves when the user is done */
  prompt() {
    return new Promise(resolve => {
      this._resolve = resolve;
      this._selected = 0;
      this._render();
      document.getElementById('feedbackModal').style.display = 'flex';
    });
  },

  /** Build the 5 star buttons */
  _render() {
    const row = document.getElementById('fbStarRow');
    if (!row) return;
    row.innerHTML = '';
    for (let i = 1; i <= 5; i++) {
      const s = document.createElement('span');
      s.className = 'fb-star';
      s.textContent = '☆';
      s.dataset.val = i;
      s.onclick = () => this._pick(i);
      row.appendChild(s);
    }
    const ta = document.getElementById('fbTextarea');
    if (ta) ta.value = '';
    this._setLabel('Tap a star to rate');
  },

  _pick(n) {
    this._selected = n;
    document.querySelectorAll('#fbStarRow .fb-star').forEach((s, idx) => {
      s.textContent = idx < n ? '★' : '☆';
    });
    const labels = ['', 'Awful', 'Bad', 'Okay', 'Good', 'Loved it!'];
    this._setLabel(labels[n]);
  },

  _setLabel(txt) {
    const el = document.getElementById('fbRatingLabel');
    if (el) { el.textContent = txt; el.style.color = '#9ca3af'; }
  },

  /** User clicked "Submit & Deploy" */
  async submit() {
    if (!this._selected) {
      const el = document.getElementById('fbRatingLabel');
      if (el) { el.textContent = '⚠️ Please pick a star first'; el.style.color = '#ef4444'; }
      return;
    }
    const text = (document.getElementById('fbTextarea') || {}).value || '';
    this._post(this._selected, text);   // fire-and-forget — don't block deploy
    this._close();
  },

  /** User clicked "Skip" */
  skip() { this._close(); },

  _close() {
    document.getElementById('feedbackModal').style.display = 'none';
    if (this._resolve) { this._resolve(); this._resolve = null; }
  },

  /** POST to backend — best-effort, never throws to the caller */
  async _post(rating, text) {
    try {
      const res = await fetch(`${API_BASE_URL}/api/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rating, text })
      });
      if (res.ok) {
        console.log('✅ Feedback saved to Firestore');
      } else {
        console.warn('⚠️ Feedback POST returned', res.status);
      }
    } catch (e) {
      console.warn('⚠️ Feedback POST failed (non-critical):', e);
    }
  }
};

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

// Route Protection - Must have CV data to access this page
(function() {
    const cvData = localStorage.getItem('cvData');
    if (!cvData || cvData === '{}' || cvData === 'null') {
        alert('Please upload your CV first to access this page.');
        window.location.href = 'index.html';
        throw new Error('CV data required');
    }
})();

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
    // ── ask for feedback before anything else ──
    await Feedback.prompt();

    const step4          = document.getElementById('step4');
    const deployBtn      = document.getElementById('deployBtn');
    const progressDiv    = document.getElementById('deploymentProgress');
    const successDiv     = document.getElementById('deploymentSuccess');
    const statusText     = document.getElementById('deploymentStatusText');

    // Hide button, show progress  — all queries scoped to step4
    if (deployBtn)   deployBtn.style.display = 'none';
    const summary    = step4 && step4.querySelector('.deployment-summary');
    const preview    = step4 && step4.querySelector('.preview-section');
    const btnGroup   = step4 && step4.querySelector('.button-group');
    if (summary)     summary.style.display  = 'none';
    if (preview)     preview.style.display  = 'none';
    if (btnGroup)    btnGroup.style.display = 'none';
    if (progressDiv) progressDiv.style.display = 'block';

    try {
        // Step 1: Creating repository
        if (statusText) statusText.textContent = 'Creating GitHub repository...';
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Step 2: Generating portfolio
        if (statusText) statusText.textContent = 'Generating your portfolio...';
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Step 3: Deploying
        if (statusText) statusText.textContent = 'Deploying to GitHub Pages...';

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
            if (progressDiv) progressDiv.style.display = 'none';
            if (successDiv)  successDiv.style.display  = 'block';

            // Build URL with safe fallback
            const portfolioUrl = (result.data && result.data.portfolio_url)
                || `https://${wizardData.githubUsername}.github.io`;

            const linkEl = document.getElementById('successPortfolioUrl');
            if (linkEl) {
                linkEl.href        = portfolioUrl;
                linkEl.textContent = portfolioUrl;
            }

            // Celebration effect
            celebrate();

        } else {
            throw new Error(result.error || 'Deployment failed');
        }

    } catch (error) {
        console.error('Deployment error:', error);

        if (progressDiv) progressDiv.style.display = 'none';

        // Show error
        alert(`Deployment failed: ${error.message}`);

        // Restore UI — all scoped to step4
        if (deployBtn) deployBtn.style.display = 'flex';
        if (summary)   summary.style.display  = 'block';
        if (preview)   preview.style.display  = 'block';
        if (btnGroup)  btnGroup.style.display = 'flex';
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