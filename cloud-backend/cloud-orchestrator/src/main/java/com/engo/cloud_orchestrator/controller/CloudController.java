package com.engo.cloudorchestrator.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.engo.cloudorchestrator.model.TelemetryRecord;
import com.engo.cloudorchestrator.service.TelemetryService;

@RestController
@RequestMapping("/api/v1/cloud")
public class CloudController {

    @Autowired private TelemetryService service; // Call the service instead of the repo

    @PostMapping("/sync")
    public String receiveData(@RequestBody TelemetryRecord record) {
        service.processTelemetry(record);
        return "✅ Cloud HQ: Data Processed for " + record.getVehicleId();
    }
}