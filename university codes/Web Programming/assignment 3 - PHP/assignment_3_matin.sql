-- phpMyAdmin SQL Dump
-- version 4.8.3
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jul 21, 2019 at 07:44 PM
-- Server version: 10.1.36-MariaDB
-- PHP Version: 7.2.10

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `assignment_3`
--

-- --------------------------------------------------------

--
-- Table structure for table `data`
--

CREATE TABLE `data` (
  `id` int(6) UNSIGNED NOT NULL,
  `title` varchar(50) COLLATE utf32_persian_ci NOT NULL,
  `content` text COLLATE utf32_persian_ci NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf32 COLLATE=utf32_persian_ci;

--
-- Dumping data for table `data`
--

INSERT INTO `data` (`id`, `title`, `content`) VALUES
(1, 'myTitle', 'dfjasd;kfj;askldjfa;lskdjf;lasdkjfasdklfjskldfjalsfkjdsladfjs'),
(8, 'ساری: رونالدو وینگر چپ خواهد بود', 'مائوریتسیو ساری گفت: \"کریستیانو رونالدو طی این سالها موفق به کسب تمامی افتخارات فردی و تیمی در دنیای فوتبال شده است و طی این مدت در جناح چپ در ترکیب قرار گرفته است. در وهله اول ما نیز در یوونتوس از رونالدو در سمت چپ استفاده خواهیم کرد اما با توجه به توانایی هایی که این بازیکن دارد، تغییر پست ده متری مشکلی برای رونالدو ایجاد نخواهد کرد'),
(9, 'ساری: رونالدو وینگر چپ خواهد بود', 'مائوریتسیو ساری گفت: \"کریستیانو رونالدو طی این سالها موفق به کسب تمامی افتخارات فردی و تیمی در دنیای فوتبال شده است و طی این مدت در جناح چپ در ترکیب قرار گرفته است. در وهله اول ما نیز در یوونتوس از رونالدو در سمت چپ استفاده خواهیم کرد اما با توجه به توانایی هایی که این بازیکن دارد، تغییر پست ده متری مشکلی برای رونالدو ایجاد نخواهد کرد'),
(10, 'asdf', 'asdfasdfasdfasdfasdfasdfasشبشسیبشسیبشسیبشسیب'),
(11, 'بورلاند سی', 'برلاند بیرلاند سی سی سی جی');

-- --------------------------------------------------------

--
-- Table structure for table `userpass`
--

CREATE TABLE `userpass` (
  `id` int(6) UNSIGNED NOT NULL,
  `username` varchar(50) COLLATE utf32_persian_ci NOT NULL,
  `password` varchar(50) COLLATE utf32_persian_ci NOT NULL,
  `admin` bit(1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf32 COLLATE=utf32_persian_ci;

--
-- Dumping data for table `userpass`
--

INSERT INTO `userpass` (`id`, `username`, `password`, `admin`) VALUES
(1, 'admin', 'admin', b'1'),
(3, 'asdaf', 'asdf', b'0'),
(4, 'asdf', 'asdf', b'0');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `data`
--
ALTER TABLE `data`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `userpass`
--
ALTER TABLE `userpass`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `data`
--
ALTER TABLE `data`
  MODIFY `id` int(6) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=12;

--
-- AUTO_INCREMENT for table `userpass`
--
ALTER TABLE `userpass`
  MODIFY `id` int(6) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=5;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
