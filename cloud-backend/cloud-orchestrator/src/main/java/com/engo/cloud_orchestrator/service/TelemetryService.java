package com.engo.cloudorchestrator.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

import com.engo.cloudorchestrator.model.TelemetryRecord;
import com.engo.cloudorchestrator.repository.TelemetryRepository;

@Service
public class TelemetryService {

    @Autowired private TelemetryRepository repository;
    @Autowired private StringRedisTemplate redis;
    @Autowired private KafkaTemplate<String, String> kafka;

    public void processTelemetry(TelemetryRecord record) {
        // 1. Save to H2 Database (Permanent History)
        repository.save(record);

        // 2. Update Redis (Live Speed for the Dashboard)
        redis.opsForValue().set("car_status:" + record.getVehicleId(), record.getHealthStatus());

        // 3. Send to Kafka (For future AI Analytics)
        kafka.send("vfc-telemetry", "Vehicle " + record.getVehicleId() + " status: " + record.getHealthStatus());
        
        System.out.println("🚀 Cloud HQ: Processed data for vehicle " + record.getVehicleId());
    }
}