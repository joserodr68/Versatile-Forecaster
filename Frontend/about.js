document.addEventListener('DOMContentLoaded', () => {
    // Function to check if element is in viewport
    function isInViewport(element) {
        const rect = element.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );
    }

    // Add smooth reveal animation when scrolling
    function revealOnScroll() {
        const elements = document.querySelectorAll('.about-content, .image-container');
        elements.forEach(element => {
            if (isInViewport(element) && !element.classList.contains('revealed')) {
                element.classList.add('revealed');
                element.style.opacity = '1';
                element.style.transform = 'translateY(0)';
            }
        });
    }

    // Initialize elements as hidden
    const elements = document.querySelectorAll('.about-content, .image-container');
    elements.forEach(element => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        element.style.transition = 'opacity 0.8s ease-out, transform 0.8s ease-out';
    });

    // Add scroll event listener
    window.addEventListener('scroll', revealOnScroll);
    
    // Initial check for elements in viewport
    revealOnScroll();

    // Handle image loading
    const aboutImage = document.getElementById('aboutImage');
    if (aboutImage) {
        aboutImage.addEventListener('load', () => {
            aboutImage.classList.add('loaded');
        });

        aboutImage.addEventListener('error', () => {
            console.error('Error loading image');
            aboutImage.style.display = 'none';
        });
    }
});