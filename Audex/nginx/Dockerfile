# NGINX

# Pull nginx image from Docker Hub
FROM nginx

# Remove the default nginx config file and replace it with the new one
RUN rm /etc/nginx/conf.d/default.conf

# Copy content from current dir to the corresponding dir in the container
#COPY nginx.conf /etc/nginx/conf.d/ # This was the originial, which didn't work
COPY nginx.conf /etc/nginx/conf.d/default.conf