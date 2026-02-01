// Load CV data
let cvData = null;
let jobDetails = null;

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
    loadData();
    generateCV();
}

function loadData() {
    try {
        cvData = JSON.parse(localStorage.getItem('cvData') || '{}');
        jobDetails = JSON.parse(localStorage.getItem('jobDetails') || '{}');
        
        if (!cvData || Object.keys(cvData).length === 0) {
            showError();
            return;
        }
    } catch (error) {
        console.error('Error loading data:', error);
        showError();
    }
}

function generateCV() {
    const container = document.getElementById('cvContainer');
    const personal = cvData.personal_info || {};
    const summary = cvData.professional_summary || '';
    const experience = cvData.experience || [];
    const education = cvData.education || [];
    const skills = cvData.skills || {};
    const projects = cvData.projects || [];
    const certifications = cvData.certifications || [];
    const awards = cvData.awards || [];
    
    let html = '';
    
    // Header
    html += `
        <div class="cv-header">
            <h1 class="cv-name">${personal.name || '[Your Name]'}</h1>
            <p class="cv-title">${jobDetails.target_job || 'Professional'}</p>
            <div class="cv-contact">
                ${personal.email ? `<span class="cv-contact-item">📧 ${personal.email}</span>` : ''}
                ${personal.phone ? `<span class="cv-contact-item">📱 ${personal.phone}</span>` : ''}
                ${personal.location ? `<span class="cv-contact-item">📍 ${personal.location}</span>` : ''}
                ${personal.linkedin ? `<span class="cv-contact-item">💼 ${personal.linkedin.replace('https://www.linkedin.com/in/', '')}</span>` : ''}
                ${personal.website ? `<span class="cv-contact-item">🌐 ${personal.website}</span>` : ''}
            </div>
        </div>
    `;
    
    // Start two-column layout
    html += '<div class="two-column-layout">';
    
    // MAIN COLUMN (Left - 2/3 width)
    html += '<div class="main-column">';
    
    // Professional Summary
    if (summary) {
        html += `
            <div class="cv-section">
                <h2 class="cv-section-title">Professional Summary</h2>
                <p class="cv-summary">${summary}</p>
            </div>
        `;
    }
    
    // Experience
    if (experience && experience.length > 0) {
        html += `<div class="cv-section">
            <h2 class="cv-section-title">Professional Experience</h2>`;
        
        experience.forEach(exp => {
            html += `
                <div class="cv-experience-item">
                    <div class="cv-item-title">${exp.title || '[Job Title]'}</div>
                    <div class="cv-item-subtitle">${exp.company || '[Company Name]'}</div>
                    <div class="cv-item-meta">
                        ${exp.start_date || ''} - ${exp.end_date || 'Present'}${exp.location ? ' • ' + exp.location : ''}
                    </div>
                    ${exp.responsibilities && exp.responsibilities.length > 0 ? `
                        <ul class="cv-responsibilities">
                            ${exp.responsibilities.filter(r => r.trim()).slice(0, 4).map(resp => `<li>${resp}</li>`).join('')}
                        </ul>
                    ` : ''}
                </div>
            `;
        });
        
        html += `</div>`;
    }
    
    // Projects (compact)
    if (projects && projects.length > 0) {
        html += `<div class="cv-section">
            <h2 class="cv-section-title">Projects</h2>
            <div class="compact-list">`;
        
        projects.forEach(proj => {
            html += `
                <div class="compact-item">
                    <span class="compact-item-title">${proj.name || '[Project]'}</span>
                    ${proj.description ? `<br><span style="font-size: 9pt;">${proj.description}</span>` : ''}
                    ${proj.technologies && proj.technologies.length > 0 ? `<br><span class="compact-item-meta">Technologies: ${proj.technologies.join(', ')}</span>` : ''}
                </div>
            `;
        });
        
        html += `</div></div>`;
    }
    
    html += '</div>'; // End main column
    
    // SIDE COLUMN (Right - 1/3 width)
    html += '<div class="side-column">';
    
    // Education
    if (education && education.length > 0) {
        html += `<div class="cv-section">
            <h2 class="cv-section-title">Education</h2>`;
        
        education.forEach(edu => {
            html += `
                <div class="cv-education-item">
                    <div class="cv-item-title" style="font-size: 10pt;">${edu.degree || '[Degree]'}</div>
                    <div class="cv-item-subtitle" style="font-size: 9pt;">${edu.institution || '[Institution]'}</div>
                    <div class="cv-item-meta">
                        ${edu.graduation_year || ''}${edu.location ? ' • ' + edu.location : ''}
                    </div>
                </div>
            `;
        });
        
        html += `</div>`;
    }
    
    // Skills (compact in sidebar)
    const hasSkills = Object.values(skills).some(arr => arr && arr.length > 0);
    if (hasSkills) {
        html += `<div class="cv-section"><h2 class="cv-section-title">Skills</h2>`;
        
        if (skills.programming_languages && skills.programming_languages.length > 0) {
            html += `
                <div class="cv-skill-category">
                    <div class="cv-skill-category-title">Programming</div>
                    <div class="cv-skill-tags">
                        ${skills.programming_languages.map(s => `<span class="cv-skill-tag">${s}</span>`).join(' ')}
                    </div>
                </div>
            `;
        }
        
        if (skills.frameworks && skills.frameworks.length > 0) {
            html += `
                <div class="cv-skill-category">
                    <div class="cv-skill-category-title">Frameworks</div>
                    <div class="cv-skill-tags">
                        ${skills.frameworks.map(s => `<span class="cv-skill-tag">${s}</span>`).join(' ')}
                    </div>
                </div>
            `;
        }
        
        if (skills.tools && skills.tools.length > 0) {
            html += `
                <div class="cv-skill-category">
                    <div class="cv-skill-category-title">Tools</div>
                    <div class="cv-skill-tags">
                        ${skills.tools.map(s => `<span class="cv-skill-tag">${s}</span>`).join(' ')}
                    </div>
                </div>
            `;
        }
        
        if (skills.databases && skills.databases.length > 0) {
            html += `
                <div class="cv-skill-category">
                    <div class="cv-skill-category-title">Databases</div>
                    <div class="cv-skill-tags">
                        ${skills.databases.map(s => `<span class="cv-skill-tag">${s}</span>`).join(' ')}
                    </div>
                </div>
            `;
        }
        
        if (skills.cloud && skills.cloud.length > 0) {
            html += `
                <div class="cv-skill-category">
                    <div class="cv-skill-category-title">Cloud</div>
                    <div class="cv-skill-tags">
                        ${skills.cloud.map(s => `<span class="cv-skill-tag">${s}</span>`).join(' ')}
                    </div>
                </div>
            `;
        }
        
        html += `</div>`;
    }
    
    // Certifications (compact)
    if (certifications && certifications.length > 0) {
        html += `<div class="cv-section">
            <h2 class="cv-section-title">Certifications</h2>
            <div class="compact-list">`;
        
        certifications.forEach(cert => {
            html += `
                <div class="compact-item">
                    <span class="compact-item-title">${cert.name || '[Certification]'}</span>
                    ${cert.issuer ? `<br><span class="compact-item-meta">${cert.issuer}</span>` : ''}
                </div>
            `;
        });
        
        html += `</div></div>`;
    }
    
    html += '</div>'; // End side column
    html += '</div>'; // End two-column layout
    
    container.innerHTML = html;
}

function showError() {
    document.getElementById('cvContainer').innerHTML = `
        <div style="text-align: center; padding: 3rem;">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#f56565" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <line x1="15" y1="9" x2="9" y2="15"/>
                <line x1="9" y1="9" x2="15" y2="15"/>
            </svg>
            <h3 style="margin-top: 1.5rem; color: #f56565;">No CV Data Found</h3>
            <p style="margin-top: 0.5rem; color: #718096;">Please upload a CV first.</p>
            <button onclick="window.location.href='index.html'" 
                    style="margin-top: 1.5rem; padding: 0.75rem 1.5rem; background: #e94560; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600;">
                Go Back
            </button>
        </div>
    `;
}

console.log('📄 Final CV Page Ready!');
console.log('CV Data:', cvData);

// ============================================================================
// PDF Download Function
// ============================================================================

async function downloadPDF() {
    const button = document.getElementById('downloadPdfBtn');
    const originalHTML = button.innerHTML;
    
    try {
        // Show loading state
        button.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation: spin 1s linear infinite;">
                <circle cx="12" cy="12" r="10"></circle>
            </svg>
            Generating PDF...
        `;
        button.disabled = true;
        
        console.log('📄 Starting PDF generation...');
        
        // Get CV container
        const cvContainer = document.getElementById('cvContainer');
        const printActions = document.querySelector('.print-actions');
        
        // Hide buttons temporarily
        if (printActions) printActions.style.display = 'none';
        
        // Generate canvas from HTML
        console.log('📸 Capturing CV as image...');
        const canvas = await html2canvas(cvContainer, {
            scale: 2,
            useCORS: true,
            logging: false,
            backgroundColor: '#ffffff',
            windowWidth: cvContainer.scrollWidth,
            windowHeight: cvContainer.scrollHeight
        });
        
        // Show buttons again
        if (printActions) printActions.style.display = 'flex';
        
        console.log('✅ Image captured, creating PDF...');
        
        // Create PDF
        const { jsPDF } = window.jspdf;
        const imgWidth = 210; // A4 width in mm
        const pageHeight = 297; // A4 height in mm
        const imgHeight = (canvas.height * imgWidth) / canvas.width;
        let heightLeft = imgHeight;
        
        const pdf = new jsPDF({
            orientation: 'portrait',
            unit: 'mm',
            format: 'a4'
        });
        
        let position = 0;
        
        // Add image to PDF (handle multiple pages if needed)
        const imgData = canvas.toDataURL('image/png');
        pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
        heightLeft -= pageHeight;
        
        // Add extra pages if content is longer
        while (heightLeft > 0) {
            position = heightLeft - imgHeight;
            pdf.addPage();
            pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
            heightLeft -= pageHeight;
        }
        
        // Generate filename
        const name = cvData?.personal_info?.name || 'CV';
        const jobTitle = jobDetails?.target_job || '';
        const cleanName = name.replace(/[^a-zA-Z0-9]/g, '_');
        const cleanJob = jobTitle.replace(/[^a-zA-Z0-9]/g, '_');
        const filename = cleanJob ? `${cleanName}_${cleanJob}_CV.pdf` : `${cleanName}_CV.pdf`;
        
        console.log('💾 Downloading PDF:', filename);
        
        // Download PDF
        pdf.save(filename);
        
        console.log('✅ PDF downloaded successfully!');
        
        // Reset button after short delay
        setTimeout(() => {
            button.innerHTML = originalHTML;
            button.disabled = false;
        }, 1000);
        
    } catch (error) {
        console.error('❌ Error generating PDF:', error);
        alert('Failed to generate PDF. Please try the following:\n\n1. Refresh the page and try again\n2. Use the Print button (Ctrl+P) and save as PDF\n3. Check if your browser blocks pop-ups');
        
        // Reset button
        button.innerHTML = originalHTML;
        button.disabled = false;
    }
}

// Add spinner animation to CSS
const style = document.createElement('style');
style.textContent = `
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
`;
document.head.appendChild(style);

// Initialize download button on DOM load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDownloadButton);
} else {
    initDownloadButton();
}

function initDownloadButton() {
    const downloadBtn = document.getElementById('downloadPdfBtn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadPDF);
        console.log('✅ PDF download button initialized');
    }
}