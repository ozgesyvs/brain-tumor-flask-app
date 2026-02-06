self.addEventListener('push', (event) => {
    const options = {
        body: event.data ? event.data.text() : 'You have a notification!',
        icon: '/static/icons/notification.png',
        badge: '/static/icons/alert.png'
    };
    event.waitUntil(
        self.registration.showNotification('New Results', options)
    );
});



self.addEventListener('notificationclick', (event) => {
    event.notification.close();
    event.waitUntil(
        clients.openWindow('/profile') // Update URL as needed
    );
});