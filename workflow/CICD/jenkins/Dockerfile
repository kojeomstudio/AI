# Use Jenkins LTS stable version (pinned for safety)
# Check for the latest LTS version and update if needed
FROM jenkins/jenkins:2.516.3-lts-jdk17

# Set timezone
ENV TZ=Asia/Seoul

# Keep the setup wizard for initial security (recommended)
# To disable: uncomment the next line
# ENV JAVA_OPTS="-Djenkins.install.runSetupWizard=false"

# Copy plugin list and install plugins
COPY plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.txt

# Expose Jenkins default internal port
EXPOSE 8080
