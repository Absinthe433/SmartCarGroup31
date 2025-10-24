/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : [PROTOCOL-V3-MODIFIED] Implements learned LIDAR logic using TIM1.
  ******************************************************************************
  * @attention
  * Copyright (c) 2025 STMicroelectronics. All rights reserved.
  * This software is licensed under terms provided AS-IS.
  *
  * V5.1 - MODIFIED by Gemini to use TIM1 instead of TIM5.
  * - Replaced UART IDLE reception with a Header-Payload DMA strategy from learned code.
  * - LIDAR reception is now on UART6: IT for a 7-byte header, then DMA for a 2600-byte payload.
  * - Removed the old angle-based scan detection ('g_scan_ready' flag).
  * - Implemented TIM1 as a scheduler for data processing and transmission.
  * - All LIDAR data processing and packet sending now occurs within the TIM1 interrupt.
  * - Replaced the array-of-structs for LIDAR points with three separate arrays
  * (Quality, angle, distance) as per the learned logic.
  * - Main loop is now decoupled from LIDAR sending tasks.
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "adc.h"
#include "dma.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"
#include "i2c.h"

/* Private includes ----------------------------------------------------------*/
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stddef.h> // Required for offsetof macro

/* Private define ------------------------------------------------------------*/
#define SEND_BEGIN_1 0x55
#define SEND_BEGIN_2 0xAA

#define SEND_BEGIN_3 0xFD
#define SEND_BEGIN_4 0xDF

#define MAX_SPEED 350
#define TURN_SPEED 280
#define DIR_STOP 0
#define DIR_FORWARD 1
#define DIR_BACKWARD 2
#define DIR_LEFT 4
#define DIR_RIGHT 5
#define WHEEL_RADIUS_M 0.0325f
#define TICKS_PER_REV 770
#define WHEEL_CIRCUMFERENCE_M (2.0f * 3.14159265f * WHEEL_RADIUS_M)

#define MPU6500_ADDR 0xD0
#define GYRO_CONFIG 0x1B
#define ACCEL_CONFIG 0x1C
#define ACCEL_XOUT_H 0x3B
#define PWR_MGMT_1 0x6B
#define WHO_AM_I 0x75
#define GYRO_FS_SEL_500DPS 0x08
#define CALIBRATION_SAMPLES 500

#define LIDAR_POINTS_PER_SCAN 520
#define LIDAR_HEADER_SIZE 7
#define LIDAR_PAYLOAD_SIZE (LIDAR_POINTS_PER_SCAN * 5) // 2600 bytes

#define DOWNLINK_HEADER_0 0xAA
#define DOWNLINK_HEADER_1 0x55
#define DOWNLINK_BUFFER_SIZE 256
#define STATUS_STOPPED 0
#define STATUS_TURNING 1
#define STATUS_MOVING_STRAIGHT 2

/* Private typedef -----------------------------------------------------------*/

// Single LIDAR point structure for sending
typedef struct __attribute__((packed)) {
    uint8_t Quality;
    uint16_t angle_q6;
    uint16_t distance_q2;
} RPLIDAR_Scan_Data_Send;

// Structure containing all LIDAR points for a scan
typedef struct __attribute__((packed)) {
    uint16_t data_count;
    RPLIDAR_Scan_Data_Send send_data[LIDAR_POINTS_PER_SCAN]; // Array of LIDAR points
} RPLIDAR_Send_Data;

// The main data packet structure for Bluetooth transmission
typedef struct __attribute__((packed)) {
    uint8_t Send_Begin_1;
    uint8_t Send_Begin_2;
 RPLIDAR_Send_Data rplidar_data;
} BLUETOOTH_Send_Data;

typedef struct __attribute__((packed)) {
    uint8_t Send_Begin_3;
    uint8_t Send_Begin_4;
    uint32_t time_us;
    uint16_t cmd_id;
    uint8_t status;
    int32_t encoder_l;
    int32_t encoder_r;
    float current_yaw;
    
} BLUETOOTH_Send_Data2;

/* Private variables ---------------------------------------------------------*/
float yaw = 0.0f;
int16_t gyro_z;
int16_t gyro_offset_z;
int16_t accel_x, accel_y, accel_z;
int16_t gyro_x, gyro_y;
int16_t accel_offset_x, accel_offset_y, accel_offset_z;
int16_t gyro_offset_x, gyro_offset_y;

// Global variables for LIDAR based on the learned logic.
uint8_t g_lidar_rx_byte;                             // Single byte buffer for IT reception
uint8_t g_lidar_header_buffer[LIDAR_HEADER_SIZE];    // Buffer to store the 7-byte header
uint8_t g_lidar_raw_buffer[LIDAR_PAYLOAD_SIZE];      // Large buffer for the DMA payload (2600 bytes)
volatile uint8_t g_lidar_rx_header_idx = 0;          // Index for header reception
volatile uint8_t g_lidar_data_ready_flag = 0;        // Flag set by UART callback when DMA is complete

// Data storage arrays, as per the learned logic's processing stage
uint8_t  g_lidar_quality[LIDAR_POINTS_PER_SCAN];
uint16_t g_lidar_angle_q6[LIDAR_POINTS_PER_SCAN];
uint16_t g_lidar_distance_q2[LIDAR_POINTS_PER_SCAN];

uint8_t g_bt_rx_byte = 0;
uint8_t g_downlink_buffer[DOWNLINK_BUFFER_SIZE];
volatile uint16_t g_downlink_buffer_idx = 0;

volatile uint8_t g_car_status = STATUS_STOPPED;
volatile uint16_t g_current_cmd_id = 0;
volatile float g_target_distance_m = 0.0f;
volatile float g_target_turn_rad = 0.0f;
volatile uint8_t g_lidar_scan_processed = 0; // Flag set when LIDAR data is parsed and ready to be sent
volatile uint8_t g_is_uart1_tx_busy = 0;     // Flag to prevent starting a new DMA TX while another is active
int32_t g_start_encoder_l = 0;
int32_t g_start_encoder_r = 0;
float g_start_yaw = 0.0f;

/* External variables --------------------------------------------------------*/
extern UART_HandleTypeDef huart1;
extern UART_HandleTypeDef huart6;
extern I2C_HandleTypeDef hi2c1;
extern TIM_HandleTypeDef htim2;
extern TIM_HandleTypeDef htim4;
extern TIM_HandleTypeDef htim1;

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);

void MPU6500_Init(I2C_HandleTypeDef* hi2c);
void MPU6500_Calibrate(I2C_HandleTypeDef* hi2c);
void MPU6500_Read_All(I2C_HandleTypeDef* hi2c);
void ControlMotor(uint8_t direction, uint32_t speed_l, uint32_t speed_r);
void Calculate_Angles(void);
void SendUplinkPacket(void);
void ParseDownlinkCommand(void);
void HandleMovementTask(void);
void SendUplinkPacket2(void);

/* Private user code ---------------------------------------------------------*/

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void) {
    HAL_Init();
    SystemClock_Config();

    MX_GPIO_Init();
    MX_DMA_Init();
    MX_TIM2_Init();
    MX_TIM3_Init();
    MX_TIM4_Init();
    MX_USART1_UART_Init();
    MX_USART6_UART_Init();
    MX_I2C1_Init();

    // Initialize and start the scheduler timer TIM1
    MX_TIM1_Init();
    HAL_TIM_Base_Start_IT(&htim1);
    MX_TIM6_Init();
    HAL_TIM_Base_Start_IT(&htim6);

    HAL_TIM_Encoder_Start(&htim2, TIM_CHANNEL_ALL);
    HAL_TIM_Encoder_Start(&htim4, TIM_CHANNEL_ALL);
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_2);
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_3);
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_4);

    HAL_UART_Receive_IT(&huart1, &g_bt_rx_byte, 1);

    MPU6500_Init(&hi2c1);
    MPU6500_Calibrate(&hi2c1);

    HAL_Delay(50);
    uint8_t startCmd[] = { 0xA5, 0x20 };
    HAL_UART_Transmit(&huart6, startCmd, sizeof(startCmd), 100);

    // Start single-byte IT to catch the LIDAR header
    HAL_UART_Receive_IT(&huart6, &g_lidar_rx_byte, 1);

    uint32_t last_imu_time = HAL_GetTick();

    while (1) {
        

        HandleMovementTask();

        // Uplink packet sending is now handled by the TIM1 interrupt
    }
}



/**
  * @brief  I2C??????.
  * @param  hi2c I2C handle.
  * @retval None
  */
void HAL_I2C_ErrorCallback(I2C_HandleTypeDef *hi2c)
{
    // ?????I2C1?????
    if (hi2c->Instance == I2C1)
    {
        // ?????????LED??,????????????
        // HAL_GPIO_TogglePin(LED_GPIO_Port, LED_Pin);

        // ???????,???????????????I2C??
        // ???? MPU6500_Read_All ???????????
        if (HAL_I2C_DeInit(hi2c) != HAL_OK)
        {
            Error_Handler();
        }
        if (HAL_I2C_Init(hi2c) != HAL_OK)
        {
            Error_Handler();
        }
    }
}

/**
  * @brief  Period elapsed callback in non-blocking mode
  * @param  htim: TIM handle
  * @retval None
  */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    // --- 调度器与通信定时器 (50Hz / 20ms) ---
    if (htim->Instance == TIM1)
    {
        static uint8_t scheduler_counter = 0;
        scheduler_counter++;

        if (g_is_uart1_tx_busy == 0)
        {
            // 当计数器达到10 (10 * 20ms = 200ms), 这是发送雷达包的窗口
            if (scheduler_counter >= 10)
            {
                scheduler_counter = 0; // 计数器清零

                for (int i = 0; i <520; i++)
            {
                int offset = i * 5;
                g_lidar_quality[i] = g_lidar_raw_buffer[offset];
                g_lidar_angle_q6[i] = (((uint16_t)(g_lidar_raw_buffer[offset + 2] << 7) + (g_lidar_raw_buffer[offset + 1] >> 1)));
                g_lidar_distance_q2[i] = (((uint16_t)(g_lidar_raw_buffer[offset + 4] << 8) + (g_lidar_raw_buffer[offset + 3])));
            }

            // --- 2. Send Data ---
            SendUplinkPacket();

            // --- 3. Reset Flag ---
            g_lidar_data_ready_flag = 0;
                
            }
            // 在200ms周期内的其他时间，都发送状态包
            else
            {
                g_is_uart1_tx_busy = 1;
                /* --- [BUG FIX] 调用正确的状态包发送函数 --- */
                SendUplinkPacket2(); // 发送20ms一次的状态包
            }
        }
    }
    // --- IMU数据生产者定时器 (50Hz / 20ms) ---
    else if (htim->Instance == TIM6) {
        MPU6500_Read_All(&hi2c1);
        Calculate_Angles();
    }
}
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1)
    {
        // When transmission is complete, free the transmitter for the next packet
        g_is_uart1_tx_busy = 0;
    }
}
/**
  * @brief  UART reception complete callback.
  * @param  huart: UART handle
  * @retval None
  */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef* huart) {
    // Bluetooth UART (USART1) - 蓝牙部分保持不变
    if (huart->Instance == USART1) {
        if (g_downlink_buffer_idx < DOWNLINK_BUFFER_SIZE) {
            g_downlink_buffer[g_downlink_buffer_idx++] = g_bt_rx_byte;
        } else {
            memmove(&g_downlink_buffer[0], &g_downlink_buffer[1], DOWNLINK_BUFFER_SIZE - 1);
            g_downlink_buffer[DOWNLINK_BUFFER_SIZE - 1] = g_bt_rx_byte;
        }
        ParseDownlinkCommand();
        HAL_UART_Receive_IT(&huart1, &g_bt_rx_byte, 1);
    }
    // LIDAR UART (USART6) - 雷达部分已修正
    else if (huart->Instance == USART6) {
        // 状态1: 通过中断接收帧头
        if (g_lidar_rx_header_idx < LIDAR_HEADER_SIZE)
        {
            g_lidar_header_buffer[g_lidar_rx_header_idx++] = g_lidar_rx_byte;
            
            if(g_lidar_rx_header_idx == LIDAR_HEADER_SIZE)
            {
                // 帧头接收完毕，启动DMA接收数据负载
                HAL_UART_Receive_DMA(&huart6, g_lidar_raw_buffer, LIDAR_PAYLOAD_SIZE);
            }
            else
            {
                // 继续等待下一个帧头字节
                HAL_UART_Receive_IT(&huart6, &g_lidar_rx_byte, 1);
            }
        }
        // 状态2: DMA传输完成时进入此分支
        else
        {
            // --- [BUG FIX #1] 立即在这里解析数据 ---
            for (int i = 0; i < LIDAR_POINTS_PER_SCAN; i++) {
                int offset = i * 5;
                g_lidar_quality[i] = g_lidar_raw_buffer[offset];
                g_lidar_angle_q6[i] = (((uint16_t)(g_lidar_raw_buffer[offset + 2] << 7) + (g_lidar_raw_buffer[offset + 1] >> 1)));
                g_lidar_distance_q2[i] = (((uint16_t)(g_lidar_raw_buffer[offset + 4] << 8) + (g_lidar_raw_buffer[offset + 3])));
            }
            
            // --- 设置正确的标志位，通知调度器雷达数据已处理完毕 ---
            g_lidar_scan_processed = 1;

            // --- 为下一帧做准备 ---
            g_lidar_rx_header_idx = 0;
            HAL_UART_Receive_IT(&huart6, &g_lidar_rx_byte, 1);
        }
    }
}

/**
  * @brief  Sends the uplink data packet containing vehicle status and LIDAR data.
  * @note   This function is called from the TIM1 interrupt.
  */
void SendUplinkPacket(void) {
    // Declare as static so it's not allocated on the stack every call
    static BLUETOOTH_Send_Data lidar_packet;
    
    // 1. Fill in the header and vehicle status data
    lidar_packet.Send_Begin_1 = SEND_BEGIN_1;
    lidar_packet.Send_Begin_2 = SEND_BEGIN_2;
    
    
    // 2. Fill in the LIDAR data from the global arrays
    lidar_packet.rplidar_data.data_count = LIDAR_POINTS_PER_SCAN;
    
    for (uint16_t i = 0; i < LIDAR_POINTS_PER_SCAN; i++) {
        lidar_packet.rplidar_data.send_data[i].Quality = g_lidar_quality[i];
        lidar_packet.rplidar_data.send_data[i].angle_q6 = g_lidar_angle_q6[i];
        lidar_packet.rplidar_data.send_data[i].distance_q2 = g_lidar_distance_q2[i];
    }
    
    
                           
    // 4. Transmit the entire packet structure via USART1 using DMA
    HAL_UART_Transmit_DMA(&huart1, (uint8_t*)&lidar_packet, sizeof(BLUETOOTH_Send_Data));
}
void SendUplinkPacket2(void) {
    // Declare as static so it's not allocated on the stack every call
    static BLUETOOTH_Send_Data2 status_packet;
    
    // 1. Fill in the header and vehicle status data
    status_packet.Send_Begin_3 = SEND_BEGIN_3;
    status_packet.Send_Begin_4 = SEND_BEGIN_4;
    status_packet.time_us = HAL_GetTick() * 1000;
    status_packet.cmd_id = g_current_cmd_id;
    status_packet.status = g_car_status;
    status_packet.encoder_l = (int32_t)__HAL_TIM_GET_COUNTER(&htim2);
    status_packet.encoder_r = (int32_t)__HAL_TIM_GET_COUNTER(&htim4);
    status_packet.current_yaw = yaw;
    
    
                           
    // 4. Transmit the entire packet structure via USART1 using DMA
    HAL_UART_Transmit_DMA(&huart1, (uint8_t*)&status_packet, sizeof(BLUETOOTH_Send_Data2));
}

void ParseDownlinkCommand(void) {
	while (g_downlink_buffer_idx >= 16) {
        if (g_downlink_buffer[0] != DOWNLINK_HEADER_0 || g_downlink_buffer[1] != DOWNLINK_HEADER_1) {
            memmove(&g_downlink_buffer[0], &g_downlink_buffer[1], g_downlink_buffer_idx - 1);
            g_downlink_buffer_idx--;
            continue;
        }
        uint16_t payload_len;
        memcpy(&payload_len, &g_downlink_buffer[2], 2);
        if (payload_len != 10) {
            memmove(&g_downlink_buffer[0], &g_downlink_buffer[2], g_downlink_buffer_idx - 2);
            g_downlink_buffer_idx -= 2;
            continue;
        }
        uint16_t received_crc;
        memcpy(&received_crc, &g_downlink_buffer[14], 2);
		if (1) { // Bypassing CRC check for now
            memcpy(&g_current_cmd_id, &g_downlink_buffer[4], 2);
            g_target_turn_rad = 0.0f;
            g_target_distance_m = 0.0f;
            ControlMotor(DIR_STOP, 0, 0);
            float turn_rad, distance_m;
            memcpy(&turn_rad, &g_downlink_buffer[6], 4);
            memcpy(&distance_m, &g_downlink_buffer[10], 4);
            g_start_encoder_l = (int32_t)__HAL_TIM_GET_COUNTER(&htim2);
            g_start_encoder_r = (int32_t)__HAL_TIM_GET_COUNTER(&htim4);
            g_start_yaw = yaw;
            if (fabsf(distance_m) > 0.001f) {
                g_target_distance_m = distance_m;
            } else if (fabsf(turn_rad) > 0.001f) {
                g_target_turn_rad = turn_rad;
            } else {
                g_car_status = STATUS_STOPPED;
            }
            memmove(&g_downlink_buffer[0], &g_downlink_buffer[16], g_downlink_buffer_idx - 16);
            g_downlink_buffer_idx -= 16;
            return;
        } else {
            memmove(&g_downlink_buffer[0], &g_downlink_buffer[1], g_downlink_buffer_idx - 1);
            g_downlink_buffer_idx--;
        }
    }
}

void ControlMotor(uint8_t direction, uint32_t speed_l, uint32_t speed_r) {
    if (speed_l > MAX_SPEED) speed_l = MAX_SPEED;
    if (speed_r > MAX_SPEED) speed_r = MAX_SPEED;
    switch (direction) {
    case DIR_STOP:
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 0);
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 0);
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, 0);
        break;
    case DIR_FORWARD:
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 0);
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, speed_l);
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, speed_r);
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, 0);
        break;
    case DIR_BACKWARD:
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, speed_l);
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 0);
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, speed_r);
        break;
    case DIR_LEFT:
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, TURN_SPEED);
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 0);
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, TURN_SPEED);
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, 0);
        break;
    case DIR_RIGHT:
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 0);
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, TURN_SPEED);
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, TURN_SPEED);
        break;
    }
}

void HandleMovementTask(void) {
    if (g_target_turn_rad != 0.0f) {
        g_car_status = STATUS_TURNING;
        
        float target_angle_deg = g_target_turn_rad * 180.0f / 3.14159265f;
        
        float angle_diff = yaw - g_start_yaw;
        if (angle_diff < -180.0f) angle_diff += 360.0f;
        if (angle_diff > 180.0f) angle_diff -= 360.0f;
        
        if (fabsf(angle_diff) < fabsf(target_angle_deg)) {
            if (target_angle_deg > 0) {
                 ControlMotor(DIR_LEFT, TURN_SPEED, TURN_SPEED);
            } else {
                 ControlMotor(DIR_RIGHT, TURN_SPEED, TURN_SPEED);
            }
        } else {
            g_target_turn_rad = 0.0f;
            ControlMotor(DIR_STOP, 0, 0);
            g_car_status = STATUS_STOPPED;
        }
        return;
    }
    
    if (g_target_distance_m != 0.0f) {
        g_car_status = STATUS_MOVING_STRAIGHT;
        float target_revolutions = fabsf(g_target_distance_m) / WHEEL_CIRCUMFERENCE_M;
        int32_t target_ticks = (int32_t)(target_revolutions * TICKS_PER_REV);
        int32_t current_ticks_l = abs((int32_t)__HAL_TIM_GET_COUNTER(&htim2) - g_start_encoder_l);
        int32_t current_ticks_r = abs((int32_t)__HAL_TIM_GET_COUNTER(&htim4) - g_start_encoder_r);
        int32_t avg_ticks = (current_ticks_l + current_ticks_r) / 2;
        
        if (avg_ticks < target_ticks) {
            if (g_target_distance_m > 0) {
                ControlMotor(DIR_FORWARD, MAX_SPEED, MAX_SPEED);
            } else {
                ControlMotor(DIR_BACKWARD, MAX_SPEED, MAX_SPEED);
            }
        } else {
            g_target_distance_m = 0.0f;
            ControlMotor(DIR_STOP, 0, 0);
            g_car_status = STATUS_STOPPED;
        }
    }
}

void MPU6500_Init(I2C_HandleTypeDef* hi2c) {
    uint8_t check, data;
    HAL_I2C_Mem_Read(hi2c, MPU6500_ADDR, WHO_AM_I, 1, &check, 1, 100);
    if (check != 0x70) { Error_Handler(); }
    data = 0x00; HAL_I2C_Mem_Write(hi2c, MPU6500_ADDR, PWR_MGMT_1, 1, &data, 1, 100); HAL_Delay(100);
    data = GYRO_FS_SEL_500DPS; HAL_I2C_Mem_Write(hi2c, MPU6500_ADDR, GYRO_CONFIG, 1, &data, 1, 100);
    data = 0x08; HAL_I2C_Mem_Write(hi2c, MPU6500_ADDR, ACCEL_CONFIG, 1, &data, 1, 100);
}

void MPU6500_Calibrate(I2C_HandleTypeDef* hi2c) {
    int32_t gyro_sum_z = 0;
    for (int i = 0; i < CALIBRATION_SAMPLES; i++) {
        uint8_t buf[2];
        HAL_I2C_Mem_Read(hi2c, MPU6500_ADDR, 0x47, 1, buf, 2, 100);
        gyro_sum_z += (int16_t)((buf[0] << 8) | buf[1]);
        HAL_Delay(3);
    }
    gyro_offset_z = gyro_sum_z / CALIBRATION_SAMPLES;
}

void MPU6500_Read_All(I2C_HandleTypeDef* hi2c) {
    uint8_t buf[14];
    HAL_StatusTypeDef status;

    // --- [新增的I2C看门狗逻辑] ---
    // 在进行任何操作前，检查I2C外设是否处于就绪状态
    if (HAL_I2C_GetState(hi2c) != HAL_I2C_STATE_READY)
    {
        // 如果I2C卡住了，执行复位流程
        HAL_I2C_DeInit(hi2c);
        HAL_I2C_Init(hi2c);
        // 您也可以在这里闪烁一个错误LED
        return; // 本次读取失败，直接返回，等待下一次中断
    }
    // ------------------------------------

    status = HAL_I2C_Mem_Read(hi2c, MPU6500_ADDR, ACCEL_XOUT_H, 1, buf, 14, 100);

    // 如果本次读取失败，也直接返回，避免使用错误数据
    if (status != HAL_OK) {
        return;
    }

    accel_x = (int16_t)((buf[0] << 8) | buf[1]);
    accel_y = (int16_t)((buf[2] << 8) | buf[3]);
    accel_z = (int16_t)((buf[4] << 8) | buf[5]);
    gyro_x = (int16_t)((buf[8] << 8) | buf[9]);
    gyro_y = (int16_t)((buf[10] << 8) | buf[11]);
    gyro_z = (int16_t)((buf[12] << 8) | buf[13]) - gyro_offset_z;
}

void Calculate_Angles(void) {
    static uint32_t last_time = 0;
    uint32_t current_time = HAL_GetTick();
    float delta_time = (last_time == 0) ? 0.02f : (current_time - last_time) / 1000.0f;
    if (delta_time <= 0) delta_time = 0.02f;
    last_time = current_time;
    float gyro_z_dps = (gyro_z / 65.5f);
    yaw += gyro_z_dps * delta_time;
    if (yaw >= 360.0f) yaw = fmodf(yaw, 360.0f);
    else if (yaw < 0.0f) yaw = 360.0f - fmodf(fabsf(yaw), 360.0f);
}

void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 180;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) { Error_Handler(); }
  if (HAL_PWREx_EnableOverDrive() != HAL_OK) { Error_Handler(); }
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;
  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK) { Error_Handler(); }
}

void Error_Handler(void) {
    __disable_irq();
    while (1) { }
}

#ifdef  USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line) {
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
}
#endif /* USE_FULL_ASSERT */