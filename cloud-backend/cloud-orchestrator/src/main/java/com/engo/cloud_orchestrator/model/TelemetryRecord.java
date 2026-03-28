package com.engo.cloudorchestrator.model;

import java.time.LocalDateTime;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.Data;

@Entity
@Table(name = "telemetry_records")
@Data
public class TelemetryRecord {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    private String vehicleId;
    private double cpuLoad;
    private double temperature;
    private String healthStatus;
    private String hazardDetected; // <--- MAKE SURE THIS IS HERE
    private String aiReasoning;
    private LocalDateTime timestamp = LocalDateTime.now();
}