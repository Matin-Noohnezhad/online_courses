<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "final_project";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// sql to create table
$sql = "create TABLE person (
id INT(6) UNSIGNED AUTO_INCREMENT PRIMARY KEY, 
username VARCHAR(50) NOT NULL,
password VARCHAR(50) NOT NULL,
email VARCHAR(50) NOT NULL,
isAdmin bit NOT NULL,
unique key unique_email(email)
)";
$sql2 = "create TABLE categories (
id INT(6) UNSIGNED AUTO_INCREMENT PRIMARY KEY, 
categoryName VARCHAR(50) NOT NULL,
unique key unique_category_name(categoryName)
)";
$sql3 = "create TABLE store (
id INT(6) UNSIGNED AUTO_INCREMENT PRIMARY KEY, 
name VARCHAR(50) NOT NULL,
password VARCHAR(50) NOT NULL,
email VARCHAR(50) NOT NULL,
unique key unique_store_name(name),
unique key unique_email(email)
)";
$sql4 = "create TABLE good (
id INT(6) UNSIGNED AUTO_INCREMENT PRIMARY KEY, 
name VARCHAR(50) NOT NULL,
categoryName VARCHAR(50) NOT NULL,
image_dir varchar(100),
price DECIMAL NOT NULL,
store_id INT(6) UNSIGNED NOT NULL,
description text, 
confirmed bit(2),
unique key unique_img_dir(image_dir),
FOREIGN KEY (store_id) REFERENCES store(id),
FOREIGN KEY (categoryName) REFERENCES categories(categoryName)
)";
//0 = seller add good
//1 = admin confirmed it
//2 = admin reject it
$sql5 = "create TABLE buying (
id INT(6) UNSIGNED AUTO_INCREMENT PRIMARY KEY, 
good_id INT(6) UNSIGNED NOT NULL,
person_id INT(6) UNSIGNED NOT NULL,
status BIT NOT NULL,
sell_Price DECIMAL ,
FOREIGN KEY (good_id) REFERENCES good(id),
FOREIGN KEY (person_id) REFERENCES person(id)
)";
//$sql = "drop table person";
//$sql2 = "drop table categories";
//$sql3 = "drop table store";
//$sql4 = "drop table good";
//$sql5 = "drop table buying";

if ($conn->query($sql) === TRUE) {
    echo "Table person created successfully";
} else {
    echo "Error creating table: " . $conn->error;
}
echo "<br>";
if ($conn->query($sql2) === TRUE) {
    echo "Table categories created successfully";
} else {
    echo "Error creating table: " . $conn->error;
}
echo "<br>";
if ($conn->query($sql3) === TRUE) {
    echo "Table store created successfully";
} else {
    echo "Error creating table: " . $conn->error;
}
echo "<br>";
if ($conn->query($sql4) === TRUE) {
    echo "Table good created successfully";
} else {
    echo "Error creating table: " . $conn->error;
}
echo "<br>";
if ($conn->query($sql5) === TRUE) {
    echo "Table buying created successfully";
} else {
    echo "Error creating table: " . $conn->error;
}

$conn->close();
?>