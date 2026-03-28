package com.engo.cloudorchestrator.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.engo.cloudorchestrator.model.TelemetryRecord;

@Repository // Ensure this is here
public interface TelemetryRepository extends JpaRepository<TelemetryRecord, Long> {
}