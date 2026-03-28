package com.engo.cloudorchestrator;

import de.codecentric.boot.admin.server.config.EnableAdminServer; // Add this import
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@EnableAdminServer  // <--- Add this line here
public class CloudOrchestratorApplication {
    public static void main(String[] args) {
        SpringApplication.run(CloudOrchestratorApplication.class, args);
    }
}