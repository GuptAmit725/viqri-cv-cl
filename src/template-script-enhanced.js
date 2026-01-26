// Enhanced Template Script with LinkedIn Job Search Integration

// Global state
let cvData = null;
let linkedInJobs = [];
let jobAnalysis = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    await loadCVData();
    await fetchLinkedInJobs();
    await analyzeJobsAndCV();
    renderAllSections();
    setupEventListeners();
});

// Load CV data from localStorage or API
async function loadCVData() {
    const loadingState = document.getElementById('loadingState');
    const contentSections = document.getElementById('contentSections');
    
    try {
        // Try to get from localStorage first
        const storedData = localStorage.getItem('cvData');
        if (storedData) {
            cvData = JSON.parse(storedData);
        } else {
            // Fallback to API call
            const response = await fetch('/api/cv/data');
            cvData = await response.json();
        }
        
        // Display target information
        document.getElementById('targetJobDisplay').textContent = cvData.targetJob || 'Not specified';
        document.getElementById('targetLocationDisplay').textContent = cvData.targetLocation || 'Not specified';
        document.getElementById('industryDisplay').textContent = cvData.industry || 'Not specified';
        document.getElementById('experienceDisplay').textContent = cvData.experienceLevel || 'Not specified';
        
    } catch (error) {
        console.error('Error loading CV data:', error);
        showError('Failed to load CV data');
    }
}

// Fetch LinkedIn jobs using the backend API
async function fetchLinkedInJobs() {
    if (!cvData || !cvData.targetJob || !cvData.targetLocation) {
        console.warn('Missing target job or location');
        return;
    }
    
    try {
        const response = await fetch('/api/jobs/linkedin/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                keywords: cvData.targetJob,
                location: cvData.targetLocation,
                experienceLevel: cvData.experienceLevel || 'mid-senior',
                limit: 10
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to fetch LinkedIn jobs');
        }
        
        linkedInJobs = await response.json();
        console.log(`Fetched ${linkedInJobs.length} LinkedIn jobs`);
        
    } catch (error) {
        console.error('Error fetching LinkedIn jobs:', error);
        // Continue with empty jobs array
        linkedInJobs = [];
    }
}

// Analyze jobs against CV data using AI
async function analyzeJobsAndCV() {
    if (!cvData || linkedInJobs.length === 0) {
        console.warn('No CV data or jobs to analyze');
        return;
    }
    
    try {
        const response = await fetch('/api/cv/analyze-jobs', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                cvData: cvData,
                jobs: linkedInJobs
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to analyze jobs');
        }
        
        jobAnalysis = await response.json();
        console.log('Job analysis completed');
        
    } catch (error) {
        console.error('Error analyzing jobs:', error);
        // Generate fallback analysis
        jobAnalysis = generateFallbackAnalysis();
    }
}

// Generate fallback analysis if API fails
function generateFallbackAnalysis() {
    const allRequirements = new Set();
    const allSkills = new Set();
    
    // Extract requirements from job descriptions
    linkedInJobs.forEach(job => {
        if (job.description) {
            // Simple keyword extraction
            const keywords = job.description.toLowerCase()
                .match(/\b[a-z]{3,}\b/g) || [];
            keywords.forEach(kw => allRequirements.add(kw));
        }
        
        if (job.skills) {
            job.skills.forEach(skill => allSkills.add(skill));
        }
    });
    
    return {
        commonRequirements: Array.from(allRequirements).slice(0, 15),
        topSkills: Array.from(allSkills).slice(0, 20),
        matchingSkills: [],
        missingSkills: [],
        recommendations: [
            'Review job descriptions and tailor your CV to match common requirements',
            'Highlight relevant experience that aligns with job postings',
            'Add quantifiable achievements to demonstrate impact'
        ]
    };
}

// Render all content sections
function renderAllSections() {
    const loadingState = document.getElementById('loadingState');
    const contentSections = document.getElementById('contentSections');
    
    // Hide loading, show content
    loadingState.style.display = 'none';
    contentSections.style.display = 'block';
    
    // Render each section
    renderOverview();
    renderLinkedInJobs();
    renderStructure();
    renderContentRecommendations();
    renderLocationInsights();
    renderIndustryInsights();
    renderActionItems();
    renderCVPreview();
}

// Render overview with stats
function renderOverview() {
    const keySkillsCount = jobAnalysis?.matchingSkills?.length || 0;
    const missingSkillsCount = jobAnalysis?.missingSkills?.length || 0;
    const actionItemsCount = jobAnalysis?.recommendations?.length || 0;
    const keywordsCount = jobAnalysis?.topSkills?.length || 0;
    
    document.getElementById('keySkillsCount').textContent = keySkillsCount;
    document.getElementById('missingSkillsCount').textContent = missingSkillsCount;
    document.getElementById('actionItemsCount').textContent = actionItemsCount;
    document.getElementById('keywordsCount').textContent = keywordsCount;
    
    // Add animated count-up effect
    animateValue('keySkillsCount', 0, keySkillsCount, 1000);
    animateValue('missingSkillsCount', 0, missingSkillsCount, 1000);
    animateValue('actionItemsCount', 0, actionItemsCount, 1000);
    animateValue('keywordsCount', 0, keywordsCount, 1000);
}

// Render LinkedIn jobs section
function renderLinkedInJobs() {
    // Add LinkedIn Jobs section after overview
    const overviewSection = document.getElementById('overview');
    
    // Create LinkedIn jobs section
    const jobsSection = document.createElement('section');
    jobsSection.id = 'linkedin-jobs';
    jobsSection.className = 'content-section';
    jobsSection.innerHTML = `
        <div class="section-icon">💼</div>
        <h2>Top LinkedIn Jobs for You</h2>
        <p class="section-intro">Based on your target role and location, here are the most relevant job opportunities:</p>
        <div id="jobsContainer"></div>
    `;
    
    overviewSection.after(jobsSection);
    
    // Update navigation
    const navList = document.querySelector('.sidebar-nav ul');
    const jobsNavItem = document.createElement('li');
    jobsNavItem.innerHTML = '<a href="#linkedin-jobs" class="nav-link">LinkedIn Jobs</a>';
    navList.insertBefore(jobsNavItem, navList.children[1]);
    
    // Render job cards
    const jobsContainer = document.getElementById('jobsContainer');
    
    if (linkedInJobs.length === 0) {
        jobsContainer.innerHTML = `
            <div class="info-card">
                <p>No jobs found. Try adjusting your target job title or location.</p>
            </div>
        `;
        return;
    }
    
    jobsContainer.innerHTML = linkedInJobs.map((job, index) => `
        <div class="job-card" data-index="${index}">
            <div class="job-header">
                <div class="job-info">
                    <h3 class="job-title">${escapeHtml(job.title)}</h3>
                    <p class="job-company">${escapeHtml(job.company)}</p>
                    <p class="job-location">${escapeHtml(job.location)}</p>
                </div>
                ${job.logo ? `<img src="${job.logo}" alt="${job.company}" class="job-logo">` : ''}
            </div>
            
            <div class="job-details">
                ${job.postedDate ? `<span class="job-meta">📅 ${job.postedDate}</span>` : ''}
                ${job.employmentType ? `<span class="job-meta">💼 ${job.employmentType}</span>` : ''}
                ${job.experienceLevel ? `<span class="job-meta">📊 ${job.experienceLevel}</span>` : ''}
            </div>
            
            ${job.description ? `
                <div class="job-description">
                    ${escapeHtml(job.description).substring(0, 200)}...
                </div>
            ` : ''}
            
            ${job.skills && job.skills.length > 0 ? `
                <div class="job-skills">
                    ${job.skills.slice(0, 5).map(skill => 
                        `<span class="skill-tag-small">${escapeHtml(skill)}</span>`
                    ).join('')}
                    ${job.skills.length > 5 ? `<span class="skill-tag-small">+${job.skills.length - 5} more</span>` : ''}
                </div>
            ` : ''}
            
            <div class="job-actions">
                <a href="${job.url}" target="_blank" rel="noopener noreferrer" class="btn-job-link">
                    View on LinkedIn
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                        <polyline points="15 3 21 3 21 9"/>
                        <line x1="10" y1="14" x2="21" y2="3"/>
                    </svg>
                </a>
                <button class="btn-analyze-job" onclick="analyzeSpecificJob(${index})">
                    Analyze Match
                </button>
            </div>
        </div>
    `).join('');
}

// Render structure recommendations
function renderStructure() {
    const structureContent = document.getElementById('structureContent');
    
    const structure = jobAnalysis?.recommendedStructure || {
        sections: [
            { name: 'Professional Summary', priority: 'High', description: 'Compelling 2-3 sentence overview' },
            { name: 'Work Experience', priority: 'High', description: 'Focus on achievements with metrics' },
            { name: 'Skills', priority: 'High', description: 'Technical and soft skills matching job requirements' },
            { name: 'Education', priority: 'Medium', description: 'Relevant degrees and certifications' },
            { name: 'Projects', priority: 'Medium', description: 'Showcase relevant work' },
            { name: 'Certifications', priority: 'Low', description: 'Industry-recognized credentials' }
        ]
    };
    
    structureContent.innerHTML = `
        <div class="structure-list">
            ${structure.sections.map(section => `
                <div class="structure-item priority-${section.priority.toLowerCase()}">
                    <div class="structure-header">
                        <h4>${section.name}</h4>
                        <span class="priority-badge ${section.priority.toLowerCase()}">${section.priority} Priority</span>
                    </div>
                    <p>${section.description}</p>
                </div>
            `).join('')}
        </div>
    `;
}

// Render content recommendations
function renderContentRecommendations() {
    const contentRecommendations = document.getElementById('contentRecommendations');
    
    const recommendations = jobAnalysis?.contentRecommendations || {
        summary: ['Highlight key achievements', 'Focus on impact with numbers'],
        experience: ['Use STAR method', 'Quantify results'],
        skills: ['Match job requirements', 'Include proficiency levels']
    };
    
    contentRecommendations.innerHTML = `
        <div class="info-card">
            <h3>📝 Professional Summary</h3>
            <ul>
                ${(recommendations.summary || []).map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        </div>
        
        <div class="info-card">
            <h3>💼 Work Experience</h3>
            <ul>
                ${(recommendations.experience || []).map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        </div>
        
        <div class="info-card">
            <h3>⚡ Skills Section</h3>
            <ul>
                ${(recommendations.skills || []).map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        </div>
        
        ${jobAnalysis?.topSkills ? `
            <div class="info-card">
                <h3>🎯 Keywords to Include (Based on Job Analysis)</h3>
                <div class="skills-grid">
                    ${jobAnalysis.topSkills.slice(0, 15).map(skill => 
                        `<span class="skill-tag">${escapeHtml(skill)}</span>`
                    ).join('')}
                </div>
            </div>
        ` : ''}
        
        ${jobAnalysis?.missingSkills && jobAnalysis.missingSkills.length > 0 ? `
            <div class="warning-box">
                <h4>⚠️ Skills Gap Analysis</h4>
                <p>Based on job requirements, consider adding these skills:</p>
                <div class="skills-grid">
                    ${jobAnalysis.missingSkills.slice(0, 10).map(skill => 
                        `<span class="skill-tag-secondary">${escapeHtml(skill)}</span>`
                    ).join('')}
                </div>
            </div>
        ` : ''}
    `;
}

// Render location insights
function renderLocationInsights() {
    const locationContent = document.getElementById('locationContent');
    
    const insights = jobAnalysis?.locationInsights || {
        marketTrends: 'Growing demand in this region',
        salaryRange: 'Competitive salary expectations',
        topCompanies: ['Tech Corp', 'Innovation Labs', 'Digital Solutions']
    };
    
    locationContent.innerHTML = `
        <div class="info-card">
            <h3>📍 Market Overview</h3>
            <p>${insights.marketTrends}</p>
            <p><strong>Salary Range:</strong> ${insights.salaryRange}</p>
        </div>
        
        ${insights.topCompanies ? `
            <div class="info-card">
                <h3>🏢 Top Hiring Companies</h3>
                <ul>
                    ${insights.topCompanies.map(company => `<li>${company}</li>`).join('')}
                </ul>
            </div>
        ` : ''}
        
        <div class="info-card">
            <h3>💡 Location-Specific Tips</h3>
            <ul>
                <li>Research local industry standards and expectations</li>
                <li>Highlight relevant local experience or connections</li>
                <li>Adapt your CV format to regional preferences</li>
            </ul>
        </div>
    `;
}

// Render industry insights
function renderIndustryInsights() {
    const industryContent = document.getElementById('industryContent');
    
    const insights = jobAnalysis?.industryInsights || {
        trends: ['Digital transformation', 'Remote work adaptation', 'Skills-based hiring'],
        keySkills: ['Leadership', 'Communication', 'Technical expertise']
    };
    
    industryContent.innerHTML = `
        <div class="info-card">
            <h3>📈 Industry Trends</h3>
            <ul>
                ${insights.trends.map(trend => `<li>${trend}</li>`).join('')}
            </ul>
        </div>
        
        <div class="info-card">
            <h3>🔑 Most Valued Skills</h3>
            <div class="skills-grid">
                ${insights.keySkills.map(skill => 
                    `<span class="skill-tag">${skill}</span>`
                ).join('')}
            </div>
        </div>
    `;
}

// Render action items
function renderActionItems() {
    const actionItemsContent = document.getElementById('actionItemsContent');
    
    const actions = jobAnalysis?.recommendations || [
        'Tailor your professional summary to match job requirements',
        'Add quantifiable achievements to experience section',
        'Optimize keywords for ATS systems',
        'Review and update skills section'
    ];
    
    // Categorize actions by priority
    const highPriority = actions.slice(0, 3);
    const mediumPriority = actions.slice(3, 6);
    const lowPriority = actions.slice(6);
    
    actionItemsContent.innerHTML = `
        ${highPriority.length > 0 ? `
            <div class="action-priority priority-high">
                <h3>🔴 High Priority</h3>
                <ul>
                    ${highPriority.map(action => `<li>${action}</li>`).join('')}
                </ul>
            </div>
        ` : ''}
        
        ${mediumPriority.length > 0 ? `
            <div class="action-priority priority-medium">
                <h3>🟡 Medium Priority</h3>
                <ul>
                    ${mediumPriority.map(action => `<li>${action}</li>`).join('')}
                </ul>
            </div>
        ` : ''}
        
        ${lowPriority.length > 0 ? `
            <div class="action-priority priority-low">
                <h3>🟢 Nice to Have</h3>
                <ul>
                    ${lowPriority.map(action => `<li>${action}</li>`).join('')}
                </ul>
            </div>
        ` : ''}
    `;
}

// Render CV preview
function renderCVPreview() {
    const cvPreviewContent = document.getElementById('cvPreviewContent');
    
    if (!cvData) {
        cvPreviewContent.innerHTML = '<p>No CV data available</p>';
        return;
    }
    
    cvPreviewContent.innerHTML = `
        <div class="cv-header">
            <h1 class="cv-name">${cvData.name || 'Your Name'}</h1>
            <p class="cv-title">${cvData.targetJob || 'Professional Title'}</p>
            <div class="cv-contact">
                ${cvData.email ? `<span>📧 ${cvData.email}</span>` : ''}
                ${cvData.phone ? `<span>📱 ${cvData.phone}</span>` : ''}
                ${cvData.location ? `<span>📍 ${cvData.location}</span>` : ''}
                ${cvData.linkedin ? `<span>💼 LinkedIn</span>` : ''}
            </div>
        </div>
    `;
}

// Setup event listeners
function setupEventListeners() {
    // Add event listener for job analysis
    document.addEventListener('click', (e) => {
        if (e.target.closest('.btn-analyze-job')) {
            e.preventDefault();
        }
    });
}

// Analyze specific job against CV
function analyzeSpecificJob(jobIndex) {
    const job = linkedInJobs[jobIndex];
    console.log('Analyzing job:', job);
    alert(`Job Analysis: ${job.title} at ${job.company}`);
}

// Animate count-up effect
function animateValue(elementId, start, end, duration) {
    const element = document.getElementById(elementId);
    const range = end - start;
    const increment = end > start ? 1 : -1;
    const stepTime = Math.abs(Math.floor(duration / range));
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if (element) {
            element.textContent = current;
        }
        if (current === end) {
            clearInterval(timer);
        }
    }, stepTime);
}

// Show error message
function showError(message) {
    const contentSections = document.getElementById('contentSections');
    if (contentSections) {
        contentSections.innerHTML = `<div class="error-box"><p>${escapeHtml(message)}</p></div>`;
    }
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}